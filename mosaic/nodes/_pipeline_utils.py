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
"""
from __future__ import annotations

from typing import Any


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
    except ImportError:
        return  # transformers 不可用，后续 from_pretrained 会自然报错

    _missing_deps: list[str] = []

    # --- T5Tokenizer ---
    t5_tok = getattr(transformers, "T5Tokenizer", None)
    if t5_tok is not None:
        # 检查是否为 Placeholder/DummyObject（缺少后端库时 transformers
        # 返回的占位类）。hasattr 可能触发 __getattribute__ 抛出
        # ImportError，需要捕获。
        try:
            is_placeholder = not hasattr(t5_tok, "from_pretrained")
        except (ImportError, AttributeError):
            is_placeholder = True
        if is_placeholder:
            _missing_deps.append("sentencepiece")
    else:
        # T5Tokenizer 不在顶层模块，尝试显式导入
        try:
            from transformers import T5Tokenizer as _  # noqa: F401
        except ImportError:
            # 在 mock 环境中 T5Tokenizer 可能不存在，这是正常的
            # 只有当 transformers 是真实安装时才报错
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

    # 第一次尝试（可能带 variant="fp16"）
    if use_fp16_variant:
        try:
            return pipeline_class.from_pretrained(
                model_name,
                variant="fp16",
                **kwargs,
            )
        except (OSError, ValueError, EnvironmentError) as exc:
            # fp16 variant 不可用，回退到 fp32
            pass
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                _build_error_message(model_name, exc)
            ) from exc

    # 第二次尝试（不带 variant 或非 fp16 场景）
    try:
        return pipeline_class.from_pretrained(model_name, **kwargs)
    except (ImportError, AttributeError, ValueError, OSError, EnvironmentError) as exc:
        raise RuntimeError(
            _build_error_message(model_name, exc)
        ) from exc


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
    try:
        return processor_class.from_pretrained(model_name, **kwargs)
    except (ImportError, AttributeError, ValueError, OSError, EnvironmentError) as exc:
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
    # 优先使用 dtype=（新版 transformers），回退 torch_dtype=（旧版兼容）
    if dtype is not None:
        try:
            return model_class.from_pretrained(model_name, dtype=dtype, **kwargs)
        except TypeError:
            return model_class.from_pretrained(
                model_name, torch_dtype=dtype, **kwargs
            )
        except (ImportError, AttributeError, ValueError, OSError, EnvironmentError) as exc:
            raise RuntimeError(
                _build_error_message(model_name, exc)
            ) from exc
    try:
        return model_class.from_pretrained(model_name, **kwargs)
    except (ImportError, AttributeError, ValueError, OSError, EnvironmentError) as exc:
        raise RuntimeError(
            _build_error_message(model_name, exc)
        ) from exc
