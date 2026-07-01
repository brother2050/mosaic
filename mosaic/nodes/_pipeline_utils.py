"""Pipeline 加载工具。

为 diffusers / transformers 的 ``from_pretrained`` 调用提供统一的错误处理、
版本诊断和兼容性修复。

核心功能：
- **T5 tokenizer 预导入**：解决 transformers 懒加载导致 diffusers 无法识别
  T5Tokenizer 组件的问题（如 CogVideoX、AudioLDM2）。
- **fp16 variant 回退**：SDXL 系列模型仓库可能不包含 fp16 变体文件，
  首次加载失败后自动回退到 fp32。
- **版本诊断**：加载失败时在错误消息中附带 diffusers/transformers 版本信息，
  并给出修复建议。
- **cache_dir 支持**：显式读取 ``HF_HOME`` 环境变量并传递给
  ``from_pretrained(cache_dir=...)``，确保自定义缓存路径对所有节点生效。
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_cache_dir() -> str | None:
    """解析 HuggingFace 缓存目录。

    优先级与 HuggingFace 库一致：
    1. ``HF_HUB_CACHE`` 环境变量
    2. ``HF_HOME`` 环境变量下的 ``hub`` 子目录（通过 :class:`MosaicEnv`）
    3. ``None``（让 HF 库使用默认路径）

    显式返回 ``cache_dir`` 而非依赖 HF 库隐式读取，确保行为可预测。
    """
    # 优先使用 HF_HUB_CACHE（最高优先级，与 HF 库行为一致）
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache and hf_hub_cache.strip():
        return hf_hub_cache.strip()

    # 通过 MosaicEnv 统一读取 HF_HOME（与项目其他模块保持一致）
    try:
        from mosaic.core.env import MosaicEnv

        hf_home = MosaicEnv.get_hf_home()
        default_home = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        if hf_home != default_home:
            return os.path.join(hf_home, "hub")
    except Exception:  # noqa: BLE001
        pass

    return None


def _preimport_t5_components() -> None:
    """预导入 T5 相关组件，避免 transformers 懒加载导致 diffusers 无法识别。

    背景：transformers 使用 ``_LazyModule`` 延迟加载子模块。当 T5Tokenizer
    所需的后端库（如 ``sentencepiece``）未安装时，transformers 不会直接抛出
    ``ImportError``，而是返回一个 ``Placeholder``/``DummyObject`` 类。该类没有
    ``from_pretrained`` 方法，导致 diffusers 抛出
    ``ValueError: cannot be loaded as it does not seem to have any loading methods``。

    解决方案：在调用 ``Pipeline.from_pretrained`` 之前，显式访问 T5Tokenizer
    类。如果检测到 Placeholder（缺少 ``from_pretrained``），则收集缺失依赖信息
    并抛出明确的 ``ImportError``，指导用户安装缺失的库。
    """
    try:
        import transformers
    except (ImportError, ValueError):
        return  # transformers 不可用或 torch.__spec__ 异常，后续 from_pretrained 会自然报错

    _missing_deps: list[str] = []

    # --- T5Tokenizer ---
    # 使用 try/import 统一检测，替代脆弱的 getattr + hasattr 组合
    # （hasattr 会触发 __getattribute__，行为受占位类影响且难以区分真实
    # 类与 Placeholder）。直接尝试导入并访问 ``from_pretrained``：真实
    # T5Tokenizer 能正常取到方法；缺 sentencepiece 等 backend 时 transformers
    # 返回的占位类在访问方法时会抛 ImportError。
    try:
        from transformers import T5Tokenizer as _t5_tok  # noqa: F401

        _ = _t5_tok.from_pretrained  # 触发占位类检测
    except (ImportError, AttributeError):
        # 导入失败或为占位类（缺少 sentencepiece 等后端库）。
        # 仅当 transformers 为真实安装时才上报缺失依赖，避免在 mock 环境
        # 中误报。
        if hasattr(transformers, "__version__"):
            _missing_deps.append("sentencepiece")

    # --- T5TokenizerFast ---
    try:
        from transformers import T5TokenizerFast  # noqa: F401
    except ImportError:
        # tokenizer（fast 版本）需要 tokenizers 库，通常已随 transformers 安装
        pass

    # --- T5EncoderModel ---
    try:
        from transformers import T5EncoderModel  # noqa: F401
    except ImportError:
        pass  # 模型加载失败不阻塞 tokenizer

    if _missing_deps:
        deps_str = " ".join(_missing_deps)
        raise ImportError(
            f"T5 tokenizer components could not be loaded because the following "
            f"dependencies are missing: {', '.join(_missing_deps)}.\n"
            f"This causes diffusers to see a Placeholder class instead of the "
            f"real T5Tokenizer, resulting in a 'cannot be loaded' error.\n"
            f"Fix: pip install {deps_str}"
        )


def _get_version_info() -> dict[str, str]:
    """获取 diffusers 和 transformers 的版本信息。"""
    info: dict[str, str] = {}
    try:
        import diffusers

        info["diffusers"] = getattr(diffusers, "__version__", "unknown")
    except ImportError:
        info["diffusers"] = "not installed"
    try:
        import transformers

        info["transformers"] = getattr(transformers, "__version__", "unknown")
    except ImportError:
        info["transformers"] = "not installed"
    return info


def _build_error_message(model_name: str, exc: Exception) -> str:
    """构建包含版本诊断信息的错误消息。"""
    versions = _get_version_info()
    # 检测常见兼容性问题
    hints: list[str] = []
    error_str = str(exc).lower()

    if "placeholder" in error_str or "cannot be loaded" in error_str:
        # Placeholder 问题通常由缺失 sentencepiece 引起
        try:
            import sentencepiece  # noqa: F401
            sp_ok = True
        except ImportError:
            sp_ok = False

        if not sp_ok:
            hints.append(
                "T5Tokenizer requires the 'sentencepiece' library which is not "
                "installed. This causes transformers to return a Placeholder "
                "class that diffusers cannot load.\n"
                "  Fix: pip install sentencepiece"
            )
        else:
            hints.append(
                "This is likely a diffusers/transformers version mismatch "
                "(T5 tokenizer lazy-loading issue). "
                "Try: pip install 'transformers>=4.44.0' 'diffusers>=0.30.0'"
            )

    if "variant" in error_str or "fp16" in error_str:
        hints.append(
            "The model repository may not have fp16 variant files. "
            "The system will retry without variant=fp16."
        )

    if "encoderdecodercache" in error_str or "cannot import name" in error_str:
        hints.append(
            "transformers/diffusers API mismatch detected. "
            "Try upgrading both: pip install -U diffusers transformers"
        )

    if not hints:
        hints.append(
            "Check that the model name/path is correct and all dependencies are installed."
        )

    version_str = ", ".join(f"{k}={v}" for k, v in versions.items())
    hint_str = "\n".join(f"  - {h}" for h in hints)

    return (
        f"Failed to load pipeline for '{model_name}'.\n"
        f"Versions: {version_str}\n"
        f"Original error: {exc}\n"
        f"Possible fixes:\n{hint_str}"
    )


def _cache_class_name(cls: Any) -> str:
    """生成用于模型缓存键的类名字符串。

    优先返回类的 ``__name__``；当传入对象没有可用的 ``__name__``（例如测试
    环境中 ``diffusers``/``transformers`` 被替换为 ``MagicMock``，而
    ``MagicMock.__name__`` 会抛出 ``AttributeError``）时，回退到 ``str(cls)``，
    保证缓存键始终为可哈希的字符串。

    返回字符串而非类对象，可绕过 :meth:`ModelCache._make_key` 对
    ``__name__`` 的访问，使缓存在 mock 环境下同样可用。
    """
    if isinstance(cls, str):
        return cls
    name = getattr(cls, "__name__", None)
    if isinstance(name, str):
        return name
    return str(cls)


def safe_load_pipeline(
    pipeline_class: Any,
    model_name: str,
    *,
    needs_t5: bool = False,
    variant_fp16: bool = False,
    dtype_str: str | None = None,
    **kwargs: Any,
) -> Any:
    """安全加载 diffusers Pipeline。

    Parameters
    ----------
    pipeline_class : type
        diffusers Pipeline 类（如 ``CogVideoXPipeline``）。
    model_name : str
        模型名称或路径（HuggingFace Hub ID 或本地路径）。
    needs_t5 : bool, 默认 False
        该 Pipeline 是否内部依赖 T5（如 CogVideoX、AudioLDM2）。
        为 True 时会在加载前预导入 T5 组件。
    variant_fp16 : bool, 默认 False
        是否尝试使用 fp16 变体。为 True 时先尝试 ``variant="fp16"``，
        失败后回退到无 variant（fp32）。
    dtype_str : str | None
        dtype 字符串（如 "float16"）。用于决定是否使用 fp16 variant。
    **kwargs : Any
        传递给 ``from_pretrained`` 的额外参数。

    Returns
    -------
    Any
        加载完成的 Pipeline 实例。

    Raises
    ------
    RuntimeError
        加载失败时抛出，包含版本诊断信息。
    """
    import torch  # type: ignore
    from mosaic.core.model_cache import model_cache

    # 显式传递 cache_dir，确保 HF_HOME 对所有节点生效
    cache_dir = _resolve_cache_dir()
    if cache_dir is not None and "cache_dir" not in kwargs:
        kwargs["cache_dir"] = cache_dir

    # 解析 torch_dtype
    torch_dtype = kwargs.pop("torch_dtype", None)
    if torch_dtype is None and dtype_str is not None:
        if dtype_str in ("float16", "fp16"):
            torch_dtype = torch.float16
        elif dtype_str in ("bfloat16", "bf16"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    # T5 预导入
    if needs_t5:
        _preimport_t5_components()

    # 决定是否使用 fp16 variant
    use_fp16_variant = variant_fp16 or (
        dtype_str is not None and dtype_str in ("float16", "fp16")
    )

    # 查询模型缓存：命中则直接返回，避免同一模型被多个节点实例重复加载。
    # ``device`` 纳入缓存键以区分加载到不同设备的实例；节点未显式传入时为
    # ``None``。``dtype_str`` 为 None 时统一记为 "default"，保证键为字符串。
    cache_dtype = dtype_str or "default"
    cache_device = kwargs.get("device")
    cache_cls = _cache_class_name(pipeline_class)
    cached = model_cache.get(cache_cls, model_name, cache_dtype, cache_device)
    if cached is not None:
        return cached

    # 第一次尝试（可能带 variant="fp16"）
    first_exc: BaseException | None = None
    pipe: Any = None
    if use_fp16_variant:
        try:
            pipe = pipeline_class.from_pretrained(
                model_name,
                variant="fp16",
                **kwargs,
            )
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                _build_error_message(model_name, exc)
            ) from exc
        except (OSError, ValueError, RuntimeError) as exc:
            # fp16 variant 不可用（文件缺失/校验失败/部分版本抛 RuntimeError），
            # 回退到无 variant 加载。此时 torch_dtype 仍为 float16，但加载的是
            # fp32 权重再转 fp16（数值可能偏离预量化 variant）。对 VAE 已有
            # _upcast_vae_fp32 兜底，但为安全起见，已知 fp16 不安全模型应转 fp32。
            first_exc = exc
            logger.warning(
                "fp16 variant load failed for %s: %s; retrying without variant.",
                model_name,
                exc,
            )

    # 第二次尝试（不带 variant 或非 fp16 场景）
    # fp16 variant 回退：torch_dtype 从 float16 降级为 float32，避免加载
    # fp32 权重后再转 fp16 引入数值偏差（对 SD 1.5 等模型尤其重要）。
    if pipe is None:
        fallback_kwargs = dict(kwargs)
        if use_fp16_variant and torch_dtype == torch.float16:
            fallback_kwargs["torch_dtype"] = torch.float32
            logger.info(
                "Retrying %s without fp16 variant; switching dtype to float32.",
                model_name,
            )
        try:
            pipe = pipeline_class.from_pretrained(model_name, **fallback_kwargs)
        except (ImportError, AttributeError, ValueError, OSError, RuntimeError) as exc:
            if first_exc is not None:
                # 两次加载均失败：同时记录 fp16 与 fp32 的错误，避免第一次失败
                # 信息丢失导致根因难以定位（见 E1）。
                logger.error(
                    "Both fp16 variant and fp32 load failed for %s. "
                    "fp16 error: %s; fp32 error: %s",
                    model_name,
                    first_exc,
                    exc,
                )
            raise RuntimeError(
                _build_error_message(model_name, exc)
            ) from exc

    # 加载成功后存入缓存（from_pretrained 抛异常时不会到达此处）
    model_cache.put(cache_cls, model_name, cache_dtype, pipe, cache_device)
    return pipe


def safe_load_processor(
    processor_class: Any,
    model_name: str,
    **kwargs: Any,
) -> Any:
    """安全加载 transformers AutoProcessor / AutoTokenizer。

    Parameters
    ----------
    processor_class : type
        transformers 类（如 ``AutoProcessor``、``AutoTokenizer``）。
    model_name : str
        模型名称或路径。
    **kwargs : Any
        传递给 ``from_pretrained`` 的额外参数。

    Returns
    -------
    Any
        加载完成的 processor/tokenizer 实例。

    Raises
    ------
    RuntimeError
        加载失败时抛出，包含版本诊断信息。
    """
    from mosaic.core.model_cache import model_cache

    try:
        # 显式传递 cache_dir，确保 HF_HOME 对所有节点生效
        cache_dir = _resolve_cache_dir()
        if cache_dir is not None and "cache_dir" not in kwargs:
            kwargs["cache_dir"] = cache_dir

        # 查询缓存：processor/tokenizer 通常无 dtype 概念，统一以 "default"
        # 作为 dtype 维度的键；device 纳入键以区分不同设备实例。
        cache_device = kwargs.get("device")
        cache_cls = _cache_class_name(processor_class)
        cached = model_cache.get(
            cache_cls, model_name, "default", cache_device
        )
        if cached is not None:
            return cached

        processor = processor_class.from_pretrained(model_name, **kwargs)
        # 加载成功后存入缓存（from_pretrained 抛异常时不会到达此处）
        model_cache.put(
            cache_cls, model_name, "default", processor, cache_device
        )
        return processor
    except (ImportError, AttributeError, ValueError, OSError) as exc:
        raise RuntimeError(
            _build_error_message(model_name, exc)
        ) from exc


def safe_load_model(
    model_class: Any,
    model_name: str,
    *,
    dtype: Any = None,
    **kwargs: Any,
) -> Any:
    """安全加载 transformers 模型，兼容 dtype/torch_dtype 参数变更。

    Parameters
    ----------
    model_class : type
        transformers 模型类（如 ``AutoModelForCausalLM``）。
    model_name : str
        模型名称或路径。
    dtype : Any, 可选
        目标 dtype（torch.float16 等）。
    **kwargs : Any
        传递给 ``from_pretrained`` 的额外参数。

    Returns
    -------
    Any
        加载完成的模型实例。

    Raises
    ------
    RuntimeError
        加载失败时抛出，包含版本诊断信息。
    """
    from mosaic.core.model_cache import model_cache

    # 优先使用 dtype=（新版 transformers），回退 torch_dtype=（旧版兼容）
    # 显式传递 cache_dir，确保 HF_HOME 对所有节点生效
    cache_dir = _resolve_cache_dir()
    if cache_dir is not None and "cache_dir" not in kwargs:
        kwargs["cache_dir"] = cache_dir

    # 查询缓存：将 dtype 转为字符串作为缓存键（与 pipeline 的 dtype_str 维度
    # 对齐），确保同 dtype 实例命中；device 纳入键以区分不同设备实例。
    cache_dtype = str(dtype) if dtype is not None else "default"
    cache_device = kwargs.get("device")
    cache_cls = _cache_class_name(model_class)
    cached = model_cache.get(cache_cls, model_name, cache_dtype, cache_device)
    if cached is not None:
        return cached

    model: Any = None
    if dtype is not None:
        try:
            model = model_class.from_pretrained(model_name, dtype=dtype, **kwargs)
        except TypeError as exc:
            # 旧版 transformers 不接受 dtype= 关键字，回退到 torch_dtype=。
            # 记录警告，避免 TypeError 被静默回退而掩盖真正的加载错误（见 E3）。
            logger.warning(
                "TypeError during model load for %s (may be dtype keyword "
                "issue): %s; retrying with torch_dtype.",
                model_name,
                exc,
            )
            model = model_class.from_pretrained(
                model_name, torch_dtype=dtype, **kwargs
            )
        except (ImportError, AttributeError, ValueError, OSError) as exc:
            raise RuntimeError(
                _build_error_message(model_name, exc)
            ) from exc
    else:
        try:
            model = model_class.from_pretrained(model_name, **kwargs)
        except (ImportError, AttributeError, ValueError, OSError) as exc:
            raise RuntimeError(
                _build_error_message(model_name, exc)
            ) from exc

    # 加载成功后存入缓存（from_pretrained 抛异常时不会到达此处）
    model_cache.put(cache_cls, model_name, cache_dtype, model, cache_device)
    return model
