# 文件路径: mosaic/nodes/audio/tts_backends/weights/cosyvoice_convert.py
"""CosyVoice 权重转换器。

将 CosyVoice 原始 checkpoint 转换为 Mosaic 标准权重格式
（safetensors + ``config.json``）。

CosyVoice 模型结构
--------------------
CosyVoice 包含以下组件：

- **text_frontend**：文本前端，其核心 LLM（Qwen2.5 / ChatGLM）权重 *不*
  参与拷贝，而是以 HuggingFace 模型路径的形式在配置中引用。转换时仅写出
  一份 ``text_frontend_config.json``，其中包含 ``{"llm_model_path": "..."}``。
- **flow_matching**：基于 Flow Matching 的声学模型（``FlowEstimator``），
  包含 estimator 主体（``estimator.*``）与条件投影
  （``text_proj.*``、``ref_proj.*``、``spk_proj.*``）。
- **speech_tokenizer**：语音 token 量化器（``quantizer.*``、``encoder.*``）。
- **speaker_encoder**：说话人编码器（``speaker_encoder.*`` / ``encoder.*``）。
- **vocoder**：HiFi-GAN 声码器（``generator.*`` / ``hifigan.*``）。

与基类 :class:`WeightConverter` 的关键差异
----------------------------------------
其他 TTS 后端（如 Fish Speech、GPT-SoVITS、ChatTTS）会将 LLM 权重一并拷贝为
safetensors；而 CosyVoice 的 LLM 体量较大且本身以 HuggingFace 格式分发，因此
*不拷贝* LLM 权重，仅在 ``config.json`` 与 ``text_frontend_config.json`` 中记录
其 HuggingFace 模型路径，由推理后端按需加载。这也是本转换器 ``COMPONENTS``
中使用 ``flow_matching`` / ``speech_tokenizer`` / ``speaker_encoder`` 而非基类
默认 ``acoustic_model`` 的原因。

权重映射说明
------------
* flow_matching：``estimator.``、``text_proj.``、``ref_proj.``、``spk_proj.``
  前缀保持不变（仅做过滤）。
* speech_tokenizer：``quantizer.``、``encoder.`` 前缀保持不变。
* speaker_encoder：``speaker_encoder.`` / ``encoder.`` 前缀去除。
* vocoder：``generator.`` / ``hifigan.`` 前缀去除。
* 判别器权重（``disc.*``、``mpd.*``、``msd.*`` 以及包含 ``discriminator`` 的
  key）一律过滤。

依赖说明
--------
``torch`` 与 ``safetensors`` 均为惰性导入，仅在加载/保存权重时需要；
模块导入本身不依赖它们，因此在未安装 torch 的环境中也可正常 import。
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any

from mosaic.nodes.audio.tts_backends.weights.converter import WeightConverter

logger = logging.getLogger(__name__)

__all__ = ["CosyVoiceWeightConverter"]


class CosyVoiceWeightConverter(WeightConverter):
    """CosyVoice 权重转换器。

    将 CosyVoice 原始 checkpoint 转换为 Mosaic 标准权重格式。

    CosyVoice 包含五个组件：

    - ``text_frontend``：文本前端（LLM 以 HuggingFace 路径引用，不拷贝权重）
    - ``flow_matching``：Flow Matching 声学模型（``FlowEstimator``）
    - ``speech_tokenizer``：语音 token 量化器
    - ``speaker_encoder``：说话人编码器
    - ``vocoder``：HiFi-GAN 声码器

    与基类 :class:`WeightConverter` 相比，CosyVoice 的关键差异在于
    ``text_frontend`` 不拷贝 LLM（Qwen2.5 / ChatGLM）权重，而是将其
    HuggingFace 模型路径写入 ``text_frontend_config.json`` 与 ``config.json``。

    Examples
    --------
    >>> converter = CosyVoiceWeightConverter()
    >>> # 转换全部组件
    >>> result = converter.convert("/path/to/cosyvoice.pt", "/output/dir")
    >>> # 指定 LLM 模型路径（覆盖默认/自动检测）
    >>> converter = CosyVoiceWeightConverter(
    ...     llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",
    ... )
    >>> result = converter.convert("/path/to/cosyvoice.pt", "/output/dir")
    >>> # 仅转换部分组件
    >>> result = converter.convert(
    ...     "/path/to/cosyvoice.pt", "/output/dir",
    ...     components=["flow_matching", "vocoder"],
    ... )
    >>> # 预览映射关系（不实际转换）
    >>> mapping = converter.dry_run("/path/to/cosyvoice.pt")
    """

    # CosyVoice 组件列表
    # 注意：text_frontend 不拷贝权重，仅引用 LLM 模型路径
    COMPONENTS = (
        "text_frontend",
        "flow_matching",
        "speech_tokenizer",
        "speaker_encoder",
        "vocoder",
    )

    # 默认 LLM 模型路径（CosyVoice 2 默认使用 Qwen2.5-0.5B-Instruct）
    _DEFAULT_LLM_MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

    # flow_matching 保留前缀（保持原名，仅过滤）
    _FLOW_MATCHING_PREFIXES = ("estimator.", "text_proj.", "ref_proj.", "spk_proj.")
    # speech_tokenizer 保留前缀（保持原名）
    _SPEECH_TOKENIZER_PREFIXES = ("quantizer.", "encoder.")
    # speaker_encoder 前缀（去除前缀）
    # 注意：speaker_encoder. 优先于 encoder. 匹配，避免前缀被错误裁剪
    _SPEAKER_ENCODER_PREFIXES = ("speaker_encoder.", "encoder.")
    # vocoder 前缀（去除前缀）
    _VOCODER_PREFIXES = ("generator.", "hifigan.")
    # 判别器前缀（一律过滤）
    _DISCRIMINATOR_PREFIXES = ("disc.", "mpd.", "msd.")

    # estimator 层号正则：estimator.{blocks|layers}.{i}.
    _ESTIMATOR_LAYER_RE = re.compile(r"^estimator\.(?:blocks|layers)\.(\d+)\.")
    # codebook 匹配正则（用于推断 speech_tokenizer 词表大小）
    _CODEBOOK_RE = re.compile(r"codebook", re.IGNORECASE)
    # codebook 索引正则（用于推断 num_codebooks）
    _CODEBOOK_INDEX_RE = re.compile(r"(\d+)[\._-]codebook", re.IGNORECASE)

    def __init__(self, llm_model_path: str | None = None) -> None:
        """初始化 CosyVoice 权重转换器。

        Parameters
        ----------
        llm_model_path : str | None
            LLM（Qwen2.5 / ChatGLM）的 HuggingFace 模型路径。若为 None，
            则在 :meth:`convert` 时尝试从源路径自动检测，检测失败则回退到
            :attr:`_DEFAULT_LLM_MODEL_PATH`。
        """
        self.llm_model_path = llm_model_path

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def convert(
        self,
        source_path: str,
        output_path: str,
        components: list[str] | None = None,
    ) -> dict[str, str]:
        """将 CosyVoice 原始 checkpoint 转换为 Mosaic 标准格式。

        转换流程：

        1. 检查 ``source_path`` 是否存在；
        2. 检测源格式（调用 :meth:`list_formats`）；
        3. 加载原始 checkpoint；
        4. 确定 LLM 模型路径（构造参数 > 源路径自动检测 > 默认值）；
        5. 根据 ``components`` 决定转换哪些组件（None 表示全部）；
        6. 对每个组件提取并映射权重：

           - text_frontend：写出 ``text_frontend_config.json``（不拷贝权重）；
           - flow_matching / speech_tokenizer / speaker_encoder / vocoder：
             提取并映射权重后保存为 safetensors；

        7. 构建并保存 ``config.json``；
        8. 返回 ``{组件名: 输出文件路径}`` 映射。

        Parameters
        ----------
        source_path : str
            原始权重文件路径或目录。
        output_path : str
            输出目录路径。
        components : list[str] | None
            要转换的组件列表，None 表示全部。
            可选值：``["text_frontend", "flow_matching", "speech_tokenizer",
            "speaker_encoder", "vocoder"]``。

        Returns
        -------
        dict[str, str]
            ``{组件名: 输出文件路径}`` 映射。其中 text_frontend 对应
            ``text_frontend_config.json``，其余组件对应各自的 ``.safetensors``。

        Raises
        ------
        FileNotFoundError
            源路径不存在。
        ValueError
            源格式不支持或组件名称无效。
        """
        # 1. 检查源路径是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源路径不存在: {source_path}")

        # 2. 检测源格式
        formats = self.list_formats(source_path)
        if not formats:
            raise ValueError(
                f"无法识别源路径格式: {source_path}。"
                f"支持的格式：.pt/.pth/.bin/.safetensors 或包含这些文件的目录。"
            )

        # 3. 加载原始 checkpoint
        state_dict = self._load_checkpoint(source_path)

        # 4. 确定 LLM 模型路径（构造参数 > 自动检测 > 默认值）
        if not self.llm_model_path:
            detected = self._detect_llm_model_path(source_path)
            self.llm_model_path = detected or self._DEFAULT_LLM_MODEL_PATH

        # 5. 确定要转换的组件
        if components is None:
            components = list(self.COMPONENTS)
        else:
            for comp in components:
                if comp not in self.COMPONENTS:
                    raise ValueError(
                        f"无效的组件名: {comp!r}，"
                        f"可选组件: {list(self.COMPONENTS)}"
                    )

        # 6. 逐组件提取、映射权重并保存
        result: dict[str, str] = {}
        for comp in components:
            if comp == "text_frontend":
                # text_frontend 不拷贝 LLM 权重，仅写出模型路径引用
                tf_path = self._save_text_frontend_config(output_path)
                result[comp] = tf_path
                continue

            comp_state, filename = self._extract_component(state_dict, comp)
            if not comp_state:
                # 该组件在 checkpoint 中无对应权重，跳过
                continue
            filepath = self._save_safetensors(comp_state, output_path, filename)
            result[comp] = filepath

        # 7. 构建并保存 config.json
        config = self._build_config(state_dict, source_path, list(result.keys()))
        self._save_config(config, output_path)

        # 8. 返回映射
        return result

    def validate(self, converted_path: str) -> bool:
        """验证转换后的权重是否完整可用。

        验证流程：

        1. 检查 ``converted_path`` 目录是否存在；
        2. 检查 ``config.json`` 是否存在且可解析，且
           ``model_type == "cosyvoice"``；
        3. 若 components 包含 text_frontend，检查 ``text_frontend_config.json``
           存在且包含非空的 ``llm_model_path``；
        4. 检查各组件的 ``.safetensors`` 文件是否存在并可加载；
        5. 加载权重，检查关键 shape 与 ``config.json`` 一致。

        Parameters
        ----------
        converted_path : str
            转换后的权重目录路径。

        Returns
        -------
        bool
            True 表示验证通过，False 表示验证失败。
        """
        import json

        # 1. 检查目录是否存在
        if not os.path.isdir(converted_path):
            return False

        # 2. 检查 config.json 是否存在且可解析
        config_path = os.path.join(converted_path, "config.json")
        if not os.path.isfile(config_path):
            return False
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        # 检查 model_type
        if config.get("model_type") != "cosyvoice":
            return False

        components = config.get("components", [])

        # 3. 检查 text_frontend_config.json
        if "text_frontend" in components:
            tf_path = os.path.join(converted_path, "text_frontend_config.json")
            if not os.path.isfile(tf_path):
                return False
            try:
                with open(tf_path, "r", encoding="utf-8") as f:
                    tf_config: dict[str, Any] = json.load(f)
            except (json.JSONDecodeError, OSError):
                return False
            if not tf_config.get("llm_model_path"):
                return False

        # 4. 检查各组件的 safetensors 文件是否存在并加载
        component_filenames = {
            "flow_matching": "flow_matching.safetensors",
            "speech_tokenizer": "speech_tokenizer.safetensors",
            "speaker_encoder": "speaker_encoder.safetensors",
            "vocoder": "vocoder.safetensors",
        }

        all_weights: dict[str, Any] = {}
        for comp, filename in component_filenames.items():
            fpath = os.path.join(converted_path, filename)
            if not os.path.isfile(fpath):
                # 未转换的组件跳过（可能是部分转换）
                continue
            try:
                from safetensors.torch import load_file  # type: ignore

                all_weights.update(load_file(fpath))
            except ImportError:
                # safetensors 未安装，无法验证权重内容
                return False
            except Exception:  # noqa: BLE001
                return False

        # 至少要有 text_frontend 引用或某个 safetensors
        has_text_frontend = "text_frontend" in components
        if not all_weights and not has_text_frontend:
            return False

        # 5. 检查 key 和 shape 与 config.json 一致

        # 检查 flow_matching_num_layers：从 estimator 层推断
        if "flow_matching_num_layers" in config:
            layer_indices = {
                int(m.group(1))
                for k in all_weights
                if (m := self._ESTIMATOR_LAYER_RE.match(k))
            }
            if layer_indices and len(layer_indices) != config["flow_matching_num_layers"]:
                return False

        # 检查 speech_tokenizer_vocab_size：从 codebook 权重推断
        if "speech_tokenizer_vocab_size" in config:
            codebook_keys = [
                k for k in all_weights if self._CODEBOOK_RE.search(k)
            ]
            if codebook_keys:
                first_codebook = all_weights[codebook_keys[0]]
                if hasattr(first_codebook, "shape"):
                    # codebook 形状一般为 (codebook_size, dim)
                    if first_codebook.shape[0] != config["speech_tokenizer_vocab_size"]:
                        return False

        return True

    # ------------------------------------------------------------------
    # 新增方法
    # ------------------------------------------------------------------

    def dry_run(
        self,
        source_path: str,
        components: list[str] | None = None,
    ) -> dict[str, dict[str, str]]:
        """预览权重映射关系，不实际转换。

        加载原始 checkpoint，对每个组件打印 ``源 key -> 目标 key`` 及其
        shape，方便用户在正式转换前确认映射是否正确。

        特别地，text_frontend 组件不涉及权重拷贝（LLM 以 HuggingFace 路径
        引用），因此其映射为空字典，并打印 LLM 模型路径引用信息。

        Parameters
        ----------
        source_path : str
            原始权重文件路径或目录。
        components : list[str] | None
            要预览的组件列表，None 表示全部。

        Returns
        -------
        dict[str, dict[str, str]]
            ``{组件名: {源 key: 目标 key}}`` 映射。

        Raises
        ------
        FileNotFoundError
            源路径不存在。
        """
        # 1. 检查源路径是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源路径不存在: {source_path}")

        # 2. 加载 checkpoint 以获取 shape 信息
        state_dict = self._load_checkpoint(source_path)

        # 确定 LLM 模型路径用于展示
        if self.llm_model_path:
            llm_path = self.llm_model_path
        else:
            detected = self._detect_llm_model_path(source_path)
            llm_path = detected or self._DEFAULT_LLM_MODEL_PATH

        # 3. 确定要预览的组件
        if components is None:
            components = list(self.COMPONENTS)
        else:
            for comp in components:
                if comp not in self.COMPONENTS:
                    raise ValueError(
                        f"无效的组件名: {comp!r}，"
                        f"可选组件: {list(self.COMPONENTS)}"
                    )

        # 4. 逐组件打印映射关系
        result: dict[str, dict[str, str]] = {}
        for comp in components:
            logger.info("\n%s\n组件: %s\n%s", "=" * 70, comp, "=" * 70)

            if comp == "text_frontend":
                # text_frontend 不拷贝权重，仅引用 LLM 模型路径
                logger.info("  (LLM 权重不拷贝，以 HuggingFace 路径引用)")
                logger.info("  llm_model_path = %s", llm_path)
                logger.info("  输出文件: text_frontend_config.json")
                result[comp] = {}
                continue

            comp_mapping: dict[str, str] = {}
            for src_key, value in state_dict.items():
                tgt_key = self._resolve_target_key(src_key, comp)
                if tgt_key is None:
                    continue

                # 获取 shape
                shape = tuple(value.shape) if hasattr(value, "shape") else None
                logger.info("  %s  ->  %s    shape=%s", src_key, tgt_key, shape)
                comp_mapping[src_key] = tgt_key

            if not comp_mapping:
                logger.info("  (无匹配权重)")
            else:
                logger.info("  共 %d 个权重", len(comp_mapping))

            result[comp] = comp_mapping

        return result

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _load_checkpoint(self, source_path: str) -> dict[str, Any]:
        """加载原始 checkpoint。

        支持以下格式：

        - ``.pt`` / ``.pth`` / ``.bin``：使用 ``torch.load`` 加载；
        - ``.safetensors``：使用 ``safetensors.load_file`` 加载；
        - 目录：合并目录下所有支持的权重文件。

        若 ``torch.load`` 返回的字典包含 ``"state_dict"`` 或 ``"model"``
        键，则自动解包。

        Parameters
        ----------
        source_path : str
            原始权重文件路径或目录。

        Returns
        -------
        dict[str, Any]
            权重 state_dict。

        Raises
        ------
        FileNotFoundError
            源路径不存在。
        ValueError
            文件格式不支持或未能加载任何权重。
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源路径不存在: {source_path}")

        state_dict: dict[str, Any] = {}

        def _unwrap(ckpt: Any) -> dict[str, Any]:
            """解包 checkpoint 字典。

            处理 ``{"state_dict": ...}`` 和 ``{"model": ...}`` 包装格式。
            """
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    return ckpt["state_dict"]
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    return ckpt["model"]
            if isinstance(ckpt, dict):
                return ckpt
            raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")

        if os.path.isdir(source_path):
            # 目录：合并所有支持的权重文件
            for fname in sorted(os.listdir(source_path)):
                fpath = os.path.join(source_path, fname)
                if not os.path.isfile(fpath):
                    continue
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file  # type: ignore

                    state_dict.update(load_file(fpath))
                elif fname.endswith((".pt", ".pth", ".bin")):
                    import torch  # type: ignore

                    ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                    state_dict.update(_unwrap(ckpt))
        elif os.path.isfile(source_path):
            if source_path.endswith(".safetensors"):
                from safetensors.torch import load_file  # type: ignore

                state_dict = load_file(source_path)
            elif source_path.endswith((".pt", ".pth", ".bin")):
                import torch  # type: ignore

                ckpt = torch.load(source_path, map_location="cpu", weights_only=False)
                state_dict = _unwrap(ckpt)
            else:
                raise ValueError(
                    f"不支持的文件格式: {source_path}。"
                    f"支持：.pt/.pth/.bin/.safetensors"
                )

        if not state_dict:
            raise ValueError(f"未能从 {source_path} 加载任何权重")

        return state_dict

    def _detect_llm_model_path(self, source_path: str) -> str | None:
        """尝试从源路径自动检测 LLM 模型路径。

        CosyVoice 通常随 checkpoint 附带配置文件（``config.json`` 或
        ``config.yaml``），其中包含 LLM 的 HuggingFace 模型路径。本方法
        在源路径为目录时尝试解析这些配置文件。

        支持的配置键（嵌套或扁平）：

        - ``llm_model_path``
        - ``llm`` -> ``model_path``
        - ``text_frontend`` -> ``llm_model_path``

        Parameters
        ----------
        source_path : str
            原始权重文件路径或目录。

        Returns
        -------
        str | None
            检测到的 LLM 模型路径；未检测到则返回 None。
        """
        import json

        # 确定搜索目录：源路径为目录时直接使用；为文件时使用其所在父目录
        # （CosyVoice 常以单个 .pt 文件 + 同级 config.yaml 形式分发）
        if os.path.isdir(source_path):
            search_dir = source_path
        elif os.path.isfile(source_path):
            search_dir = os.path.dirname(os.path.abspath(source_path))
        else:
            return None

        candidates: list[str] = []
        for fname in os.listdir(search_dir):
            if fname in ("config.json", "config.yaml", "config.yml"):
                candidates.append(os.path.join(search_dir, fname))

        def _extract_path(obj: Any) -> str | None:
            """递归查找 LLM 模型路径。"""
            if isinstance(obj, dict):
                if isinstance(obj.get("llm_model_path"), str):
                    return obj["llm_model_path"]
                llm = obj.get("llm")
                if isinstance(llm, dict) and isinstance(llm.get("model_path"), str):
                    return llm["model_path"]
                tf = obj.get("text_frontend")
                if isinstance(tf, dict) and isinstance(tf.get("llm_model_path"), str):
                    return tf["llm_model_path"]
                for v in obj.values():
                    found = _extract_path(v)
                    if found:
                        return found
            return None

        for cpath in candidates:
            try:
                if cpath.endswith(".json"):
                    with open(cpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    # YAML 配置：惰性导入，未安装则跳过
                    try:
                        import yaml  # type: ignore
                    except ImportError:
                        continue
                    with open(cpath, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
            except (OSError, ValueError):
                continue
            found = _extract_path(data)
            if found:
                return found

        return None

    def _save_text_frontend_config(self, output_path: str) -> str:
        """写出 text_frontend 组件配置文件。

        CosyVoice 的 text_frontend 不拷贝 LLM 权重，仅记录 LLM 的
        HuggingFace 模型路径引用。本方法将
        ``{"llm_model_path": ...}`` 写入 ``text_frontend_config.json``。

        Parameters
        ----------
        output_path : str
            输出目录路径。

        Returns
        -------
        str
            配置文件路径。
        """
        import json

        os.makedirs(output_path, exist_ok=True)
        config_path = os.path.join(output_path, "text_frontend_config.json")
        config = {"llm_model_path": self.llm_model_path}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config_path

    def _extract_flow_matching_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取 Flow Matching（FlowEstimator）权重。

        提取 key 以 ``estimator.``、``text_proj.``、``ref_proj.``、
        ``spk_proj.`` 开头的权重，保持原名不变（仅过滤）。判别器权重
        （``disc.*``、``mpd.*``、``msd.*``）一律过滤。

        Flow Matching 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            Flow Matching 权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            if self._is_discriminator(key):
                continue
            if any(key.startswith(p) for p in self._FLOW_MATCHING_PREFIXES):
                extracted[key] = value
        return extracted

    def _extract_speech_tokenizer_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取语音 token 量化器权重。

        提取 key 以 ``quantizer.``、``encoder.`` 开头的权重，保持原名不变。
        判别器权重一律过滤。

        Speech Tokenizer 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            语音量化器权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            if self._is_discriminator(key):
                continue
            if any(key.startswith(p) for p in self._SPEECH_TOKENIZER_PREFIXES):
                extracted[key] = value
        return extracted

    def _extract_speaker_encoder_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取说话人编码器权重。

        提取 key 以 ``speaker_encoder.`` 或 ``encoder.`` 开头的权重，并去除
        该前缀。判别器权重一律过滤。

        注意：``speaker_encoder.`` 优先于 ``encoder.`` 匹配，以避免
        ``speaker_encoder.encoder.*`` 中的 ``encoder.`` 被错误裁剪。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            去除前缀后的说话人编码器权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            if self._is_discriminator(key):
                continue
            for prefix in self._SPEAKER_ENCODER_PREFIXES:
                if key.startswith(prefix):
                    extracted[key[len(prefix):]] = value
                    break
        return extracted

    def _extract_vocoder_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取 HiFi-GAN 声码器权重。

        提取 key 以 ``generator.`` 或 ``hifigan.`` 开头的权重，并去除该前缀。
        判别器权重（``disc.*``、``mpd.*``、``msd.*`` 以及包含
        ``discriminator`` 的 key）一律过滤。

        Vocoder 权重去除前缀后保存。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            去除前缀后的 HiFi-GAN generator 权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            if self._is_discriminator(key) or "discriminator" in key:
                continue
            for prefix in self._VOCODER_PREFIXES:
                if key.startswith(prefix):
                    extracted[key[len(prefix):]] = value
                    break
        return extracted

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _is_discriminator(self, key: str) -> bool:
        """判断 key 是否属于判别器权重。

        判别器前缀：``disc.``、``mpd.``、``msd.``。这些权重仅训练时使用，
        推理时不需要，应一律过滤。

        Parameters
        ----------
        key : str
            权重 key。

        Returns
        -------
        bool
            True 表示该 key 属于判别器，应过滤。
        """
        return any(key.startswith(p) for p in self._DISCRIMINATOR_PREFIXES)

    def _extract_component(
        self, state_dict: dict[str, Any], component: str
    ) -> tuple[dict[str, Any], str]:
        """提取并映射指定组件的权重。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。
        component : str
            组件名称。

        Returns
        -------
        tuple[dict[str, Any], str]
            ``(映射后的权重字典, 输出文件名)``。text_frontend 返回空字典与
            ``text_frontend_config.json``（实际写出逻辑在 :meth:`convert`
            中单独处理）。

        Raises
        ------
        ValueError
            组件名称无效。
        """
        if component == "text_frontend":
            # text_frontend 不含权重，由 convert 单独写出配置
            return {}, "text_frontend_config.json"
        elif component == "flow_matching":
            return (
                self._extract_flow_matching_weights(state_dict),
                "flow_matching.safetensors",
            )
        elif component == "speech_tokenizer":
            return (
                self._extract_speech_tokenizer_weights(state_dict),
                "speech_tokenizer.safetensors",
            )
        elif component == "speaker_encoder":
            return (
                self._extract_speaker_encoder_weights(state_dict),
                "speaker_encoder.safetensors",
            )
        elif component == "vocoder":
            return (
                self._extract_vocoder_weights(state_dict),
                "vocoder.safetensors",
            )
        else:
            raise ValueError(f"无效的组件名: {component!r}")

    def _resolve_target_key(self, src_key: str, component: str) -> str | None:
        """解析单个源 key 在指定组件中的目标 key。

        用于 :meth:`dry_run` 中逐 key 展示映射关系。

        映射规则：

        - text_frontend：无权重（返回 None）；
        - flow_matching：保留前缀（恒等映射）；
        - speech_tokenizer：保留前缀（恒等映射）；
        - speaker_encoder：去除 ``speaker_encoder.`` / ``encoder.`` 前缀；
        - vocoder：去除 ``generator.`` / ``hifigan.`` 前缀；
        - 判别器权重一律返回 None。

        Parameters
        ----------
        src_key : str
            源权重 key。
        component : str
            组件名称。

        Returns
        -------
        str | None
            目标 key；若该 key 不属于此组件则返回 None。
        """
        # 判别器权重一律不属于任何组件
        if self._is_discriminator(src_key) or "discriminator" in src_key:
            return None

        if component == "text_frontend":
            # text_frontend 不拷贝权重
            return None

        elif component == "flow_matching":
            # 保留前缀（恒等映射，仅过滤）
            if any(src_key.startswith(p) for p in self._FLOW_MATCHING_PREFIXES):
                return src_key
            return None

        elif component == "speech_tokenizer":
            # 保留前缀（恒等映射，仅过滤）
            if any(src_key.startswith(p) for p in self._SPEECH_TOKENIZER_PREFIXES):
                return src_key
            return None

        elif component == "speaker_encoder":
            # 去除前缀
            for prefix in self._SPEAKER_ENCODER_PREFIXES:
                if src_key.startswith(prefix):
                    return src_key[len(prefix):]
            return None

        elif component == "vocoder":
            # 去除前缀
            for prefix in self._VOCODER_PREFIXES:
                if src_key.startswith(prefix):
                    return src_key[len(prefix):]
            return None

        return None

    def _build_config(
        self,
        state_dict: dict[str, Any],
        source_path: str,
        converted_components: list[str],
    ) -> dict[str, Any]:
        """从 state_dict 中提取模型配置。

        自动推断以下配置项：

        - ``flow_matching_hidden_size``：从 estimator 的 norm/embedding 推断；
        - ``flow_matching_num_layers``：estimator（blocks/layers）层数；
        - ``speech_tokenizer_vocab_size``：从 quantizer codebook 推断；
        - ``num_codebooks``：quantizer codebook 数量；
        - ``speaker_encoder_hidden_size``：从 speaker encoder 权重推断；
        - ``llm_model_path``：LLM 的 HuggingFace 模型路径引用；
        - ``llm_backend``：固定为 ``"huggingface"``；
        - ``text_frontend_config``：指向 ``text_frontend_config.json``。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。
        source_path : str
            原始来源路径。
        converted_components : list[str]
            实际转换的组件列表。

        Returns
        -------
        dict[str, Any]
            配置字典。
        """
        config: dict[str, Any] = {
            "model_type": "cosyvoice",
            "version": "0.1",
            "source": source_path,
            "conversion_date": datetime.now().isoformat(),
            "components": converted_components,
            # LLM 权重不拷贝，仅以 HuggingFace 路径引用
            "llm_model_path": self.llm_model_path,
            "llm_backend": "huggingface",
            "text_frontend_config": "text_frontend_config.json",
        }

        # flow_matching_num_layers：从 estimator 层推断
        layer_indices: set[int] = set()
        for key in state_dict:
            match = self._ESTIMATOR_LAYER_RE.match(key)
            if match:
                layer_indices.add(int(match.group(1)))
        if layer_indices:
            config["flow_matching_num_layers"] = len(layer_indices)

        # flow_matching_hidden_size：从 estimator 的 LayerNorm 推断
        # 优先匹配 norm 类一维权重（长度最可靠地等于 hidden_size）；
        # time_embed / label_embed 等可能是不同的投影维度，故不优先。
        hidden_size: int | None = None
        for key, value in state_dict.items():
            if not key.startswith("estimator."):
                continue
            if not hasattr(value, "shape") or len(value.shape) != 1:
                continue
            if "norm" in key.lower():
                hidden_size = int(value.shape[0])
                break
        if hidden_size is None:
            # 回退：取第一个 estimator 一维权重
            for key, value in state_dict.items():
                if (
                    key.startswith("estimator.")
                    and hasattr(value, "shape")
                    and len(value.shape) == 1
                ):
                    hidden_size = int(value.shape[0])
                    break
        if hidden_size is not None:
            config["flow_matching_hidden_size"] = hidden_size

        # speech_tokenizer_vocab_size & num_codebooks：从 quantizer codebook 推断
        codebook_keys = [
            k
            for k in state_dict
            if k.startswith("quantizer.") and self._CODEBOOK_RE.search(k)
        ]
        if codebook_keys:
            first_codebook = state_dict[codebook_keys[0]]
            if hasattr(first_codebook, "shape"):
                # codebook 形状一般为 (codebook_size, dim)
                config["speech_tokenizer_vocab_size"] = int(first_codebook.shape[0])

            # num_codebooks：统计不同 codebook 索引（尽力推断）
            cb_indices: set[int] = set()
            for k in codebook_keys:
                m = self._CODEBOOK_INDEX_RE.search(k)
                if m:
                    cb_indices.add(int(m.group(1)))
            if cb_indices:
                config["num_codebooks"] = len(cb_indices)

        # speaker_encoder_hidden_size：从 speaker_encoder 一维权重推断
        se_hidden: int | None = None
        for key, value in state_dict.items():
            if (
                key.startswith("speaker_encoder.")
                and hasattr(value, "shape")
                and len(value.shape) == 1
            ):
                se_hidden = int(value.shape[0])
                break
        if se_hidden is not None:
            config["speaker_encoder_hidden_size"] = se_hidden

        return config
