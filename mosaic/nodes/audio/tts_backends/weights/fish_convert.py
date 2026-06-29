# 文件路径: mosaic/nodes/audio/tts_backends/weights/fish_convert.py
"""Fish Speech 权重转换器。

将 Fish Speech 原始 checkpoint 转换为 Mosaic 标准权重格式
（safetensors + ``config.json``）。

Fish Speech 模型结构
--------------------
Fish Speech 包含以下组件：

- **text_frontend**：统一词表嵌入层（``model.embed_tokens.weight``）
- **acoustic_model**：LLaMA 自回归模型，映射为 ``LlamaForCausalLM`` 格式
- **vocoder**：HiFi-GAN generator 声码器，将 latent 转换为波形
- **vq_decoder**：VQ 解码器，将量化 latent 解码为连续特征
- **audio_encoder**：音频编码器（VQGAN encoder / DAC），将音频编码为 latent

与基类 :class:`WeightConverter` 相比，Fish Speech 额外包含 VQ Decoder
和 AudioEncoder 组件，因此 ``COMPONENTS`` 中增加了 ``"vq_decoder"`` 与
``"audio_encoder"``。

权重映射说明
------------
Fish Speech 的 LLaMA 模型权重已经使用 ``LlamaForCausalLM`` 的标准命名
（``model.embed_tokens.*``、``model.layers.{i}.*``、``model.norm.*``、
``lm_head.*``），因此 :attr:`FISH_TO_LLAMA_MAP` 中的映射基本为恒等映射。
由于 Fish Speech 使用统一词表，``embed_tokens`` 直接对应，无需额外处理。

text_frontend 组件提取 ``model.embed_tokens.weight``（统一词表嵌入），
acoustic_model 组件提取全部 LLaMA 权重（含 ``embed_tokens``）。
vocoder、vq_decoder、audio_encoder 组件的权重保持原名不变。

依赖说明
--------
``torch`` 与 ``safetensors`` 均为惰性导入，仅在加载/保存权重时需要；
模块导入本身不依赖它们。
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any

from mosaic.nodes.audio.tts_backends.weights.converter import WeightConverter

__all__ = ["FishWeightConverter"]


class FishWeightConverter(WeightConverter):
    """Fish Speech 权重转换器。

    将 Fish Speech 原始 checkpoint 转换为 Mosaic 标准权重格式。

    Fish Speech 包含五个组件：

    - ``text_frontend``：统一词表嵌入层（``model.embed_tokens.weight``）
    - ``acoustic_model``：LLaMA 自回归模型（映射为 ``LlamaForCausalLM`` 格式）
    - ``vocoder``：HiFi-GAN generator 声码器
    - ``vq_decoder``：VQ 解码器
    - ``audio_encoder``：音频编码器（VQGAN encoder / DAC）

    与基类 :class:`WeightConverter` 相比，Fish Speech 额外包含 VQ Decoder
    和 AudioEncoder 组件，因此 ``COMPONENTS`` 中增加了 ``"vq_decoder"``
    与 ``"audio_encoder"``。

    Examples
    --------
    >>> converter = FishWeightConverter()
    >>> # 转换全部组件
    >>> result = converter.convert("/path/to/fish.pt", "/output/dir")
    >>> # 仅转换部分组件
    >>> result = converter.convert(
    ...     "/path/to/fish.pt", "/output/dir",
    ...     components=["acoustic_model", "vocoder"],
    ... )
    >>> # 预览映射关系（不实际转换）
    >>> mapping = converter.dry_run("/path/to/fish.pt")
    """

    # Fish Speech 组件列表（在基类基础上增加 vq_decoder 与 audio_encoder）
    COMPONENTS = (
        "text_frontend",
        "acoustic_model",
        "vocoder",
        "vq_decoder",
        "audio_encoder",
    )

    # Fish Speech LLaMA 权重 → LlamaForCausalLM 权重映射
    # Fish Speech 使用统一词表，embed_tokens 直接对应（恒等映射）
    # 静态映射（不含层号）；带层号的通配映射在 _map_llama_weights 中处理
    FISH_TO_LLAMA_MAP: dict[str, str] = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        # 每一层 Transformer（带层号的通配映射，在 _map_llama_weights 中处理）：
        # model.layers.{i}.self_attn.q_proj.weight          → model.layers.{i}.self_attn.q_proj.weight
        # model.layers.{i}.self_attn.k_proj.weight          → model.layers.{i}.self_attn.k_proj.weight
        # model.layers.{i}.self_attn.v_proj.weight          → model.layers.{i}.self_attn.v_proj.weight
        # model.layers.{i}.self_attn.o_proj.weight          → model.layers.{i}.self_attn.o_proj.weight
        # model.layers.{i}.mlp.gate_proj.weight             → model.layers.{i}.mlp.gate_proj.weight
        # model.layers.{i}.mlp.down_proj.weight             → model.layers.{i}.mlp.down_proj.weight
        # model.layers.{i}.mlp.up_proj.weight               → model.layers.{i}.mlp.up_proj.weight
        # model.layers.{i}.input_layernorm.weight           → model.layers.{i}.input_layernorm.weight
        # model.layers.{i}.post_attention_layernorm.weight  → model.layers.{i}.post_attention_layernorm.weight
    }

    # VQ Decoder 权重保持原名
    # HiFi-GAN 权重保持原名（只保留 generator 部分，排除 discriminator）
    # AudioEncoder 权重保持原名

    # LLaMA 层号匹配正则：model.layers.{i}.{suffix}
    _LLAMA_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    # 统一词表嵌入 key
    _EMBED_KEY = "model.embed_tokens.weight"
    # codebook 匹配正则（用于推断 audio_vocab_size）
    _CODEBOOK_RE = re.compile(r"codebook", re.IGNORECASE)

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def convert(
        self,
        source_path: str,
        output_path: str,
        components: list[str] | None = None,
    ) -> dict[str, str]:
        """将 Fish Speech 原始 checkpoint 转换为 Mosaic 标准格式。

        转换流程：

        1. 检查 ``source_path`` 是否存在；
        2. 检测源格式（调用 :meth:`list_formats`）；
        3. 加载原始 checkpoint；
        4. 根据 ``components`` 决定转换哪些组件（None 表示全部）；
        5. 对每个组件提取并映射权重；
        6. 保存为 safetensors 格式；
        7. 保存 ``config.json``；
        8. 返回 ``{组件名: 输出文件路径}`` 映射。

        Parameters
        ----------
        source_path : str
            原始权重文件路径或目录。
        output_path : str
            输出目录路径。
        components : list[str] | None
            要转换的组件列表，None 表示全部。
            可选值：``["text_frontend", "acoustic_model", "vocoder",
            "vq_decoder", "audio_encoder"]``。

        Returns
        -------
        dict[str, str]
            ``{组件名: 输出文件路径}`` 映射。

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

        # 4. 确定要转换的组件
        if components is None:
            components = list(self.COMPONENTS)
        else:
            for comp in components:
                if comp not in self.COMPONENTS:
                    raise ValueError(
                        f"无效的组件名: {comp!r}，"
                        f"可选组件: {list(self.COMPONENTS)}"
                    )

        # 5 & 6. 逐组件提取、映射权重并保存为 safetensors
        result: dict[str, str] = {}
        for comp in components:
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
        2. 检查 ``config.json`` 是否存在且可解析；
        3. 检查各组件的 ``.safetensors`` 文件是否存在；
        4. 加载权重，检查 key 和 shape 是否与 ``config.json`` 一致。

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
        if config.get("model_type") != "fish_speech":
            return False

        # 3. 检查各组件的 safetensors 文件是否存在并加载
        component_filenames = {
            "text_frontend": "text_frontend.safetensors",
            "acoustic_model": "acoustic_model.safetensors",
            "vocoder": "vocoder.safetensors",
            "vq_decoder": "vq_decoder.safetensors",
            "audio_encoder": "audio_encoder.safetensors",
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
            except Exception:
                return False

        if not all_weights:
            return False

        # 4. 检查 key 和 shape 与 config.json 一致

        # 检查 text_vocab_size & hidden_size：从 embed_tokens 推断
        embed_key = "model.embed_tokens.weight"
        if embed_key in all_weights:
            embed = all_weights[embed_key]
            if "text_vocab_size" in config and hasattr(embed, "shape"):
                if embed.shape[0] != config["text_vocab_size"]:
                    return False
            if "hidden_size" in config and hasattr(embed, "shape"):
                if embed.shape[1] != config["hidden_size"]:
                    return False

        # 检查 num_layers
        layer_keys = {
            int(m.group(1))
            for k in all_weights
            if (m := re.match(r"^model\.layers\.(\d+)\.", k))
        }
        if layer_keys and "num_layers" in config:
            if len(layer_keys) != config["num_layers"]:
                return False

        # 检查 audio_vocab_size：从 codebook 权重推断
        codebook_keys = [k for k in all_weights if self._CODEBOOK_RE.search(k)]
        if codebook_keys and "audio_vocab_size" in config:
            first_codebook = all_weights[codebook_keys[0]]
            if hasattr(first_codebook, "shape"):
                # codebook 形状一般为 (num_codebooks, codebook_size) 或 (codebook_size, dim)
                # 取第一维作为 audio_vocab_size 的近似校验
                if first_codebook.shape[0] != config["audio_vocab_size"]:
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

        加载原始 checkpoint，对每个组件打印 ``源 key → 目标 key`` 及其
        shape，方便用户在正式转换前确认映射是否正确。

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
            print(f"\n{'=' * 70}")
            print(f"组件: {comp}")
            print(f"{'=' * 70}")

            comp_mapping: dict[str, str] = {}
            for src_key, value in state_dict.items():
                tgt_key = self._resolve_target_key(src_key, comp)
                if tgt_key is None:
                    continue

                # 获取 shape
                shape = tuple(value.shape) if hasattr(value, "shape") else None
                print(f"  {src_key}  →  {tgt_key}    shape={shape}")
                comp_mapping[src_key] = tgt_key

            if not comp_mapping:
                print("  (无匹配权重)")
            else:
                print(f"  共 {len(comp_mapping)} 个权重")

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

                    ckpt = torch.load(fpath, map_location="cpu")
                    state_dict.update(_unwrap(ckpt))
        elif os.path.isfile(source_path):
            if source_path.endswith(".safetensors"):
                from safetensors.torch import load_file  # type: ignore

                state_dict = load_file(source_path)
            elif source_path.endswith((".pt", ".pth", ".bin")):
                import torch  # type: ignore

                ckpt = torch.load(source_path, map_location="cpu")
                state_dict = _unwrap(ckpt)
            else:
                raise ValueError(
                    f"不支持的文件格式: {source_path}。"
                    f"支持：.pt/.pth/.bin/.safetensors"
                )

        if not state_dict:
            raise ValueError(f"未能从 {source_path} 加载任何权重")

        return state_dict

    def _map_llama_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """按 :attr:`FISH_TO_LLAMA_MAP` 映射 LLaMA 权重为 ``LlamaForCausalLM`` 格式。

        处理两类映射：

        1. **静态映射**：``model.embed_tokens.weight``、``model.norm.weight``、
           ``lm_head.weight`` 等显式列出的 key（恒等映射）；
        2. **带层号的通配映射**：``model.layers.{i}.xxx`` →
           ``model.layers.{i}.xxx``（恒等映射，同时过滤出层权重）。

        由于 Fish Speech 的 LLaMA 权重已经使用标准命名，映射为恒等映射；
        本方法主要作用是从原始 state_dict 中过滤出 acoustic_model 所需的
        LLaMA 权重。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            映射后的权重字典（key 为 LlamaForCausalLM 格式）。
        """
        mapped: dict[str, Any] = {}

        for key, value in state_dict.items():
            # 1. 静态映射（embed_tokens、norm、lm_head）
            if key in self.FISH_TO_LLAMA_MAP:
                mapped[self.FISH_TO_LLAMA_MAP[key]] = value
                continue

            # 2. 带层号的通配映射：model.layers.{i}.{suffix}
            match = self._LLAMA_LAYER_RE.match(key)
            if match:
                layer_idx, suffix = match.group(1), match.group(2)
                mapped[f"model.layers.{layer_idx}.{suffix}"] = value
                continue

        return mapped

    def _extract_embed_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """提取统一词表嵌入权重。

        提取 ``model.embed_tokens.weight``（Fish Speech 使用统一词表，
        text 与 audio 共享嵌入）。权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            映射后的嵌入权重字典。
        """
        mapped: dict[str, Any] = {}
        if self._EMBED_KEY in state_dict:
            mapped[self._EMBED_KEY] = state_dict[self._EMBED_KEY]
        return mapped

    def _extract_vocoder_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """提取 HiFi-GAN generator 声码器权重。

        提取 key 中包含 ``"generator"`` 或 ``"hifi"`` 的权重，
        并过滤掉 discriminator 权重（key 中包含 ``"discriminator"``）。

        Vocoder 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            HiFi-GAN generator 权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            # 包含 "generator" 或 "hifi"，但排除 "discriminator"
            if ("generator" in key or "hifi" in key) and "discriminator" not in key:
                extracted[key] = value
        return extracted

    def _extract_vq_decoder_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """提取 VQ 解码器权重。

        提取 key 中包含 ``"vq"`` 或 ``"decoder"`` 的权重。

        VQ Decoder 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            VQ decoder 权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            # 包含 "vq" 或 "decoder"
            if "vq" in key or "decoder" in key:
                extracted[key] = value
        return extracted

    def _extract_audio_encoder_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取音频编码器权重。

        提取 key 中包含 ``"encoder"`` 或 ``"dac"`` 的权重
        （VQGAN encoder 或 DAC 音频编码器）。

        AudioEncoder 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            音频编码器权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            # 包含 "encoder" 或 "dac"
            if "encoder" in key or "dac" in key:
                extracted[key] = value
        return extracted

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

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
            ``(映射后的权重字典, safetensors 文件名)``。
        """
        if component == "text_frontend":
            return (
                self._extract_embed_weights(state_dict),
                "text_frontend.safetensors",
            )
        elif component == "acoustic_model":
            return (
                self._map_llama_weights(state_dict),
                "acoustic_model.safetensors",
            )
        elif component == "vocoder":
            return (
                self._extract_vocoder_weights(state_dict),
                "vocoder.safetensors",
            )
        elif component == "vq_decoder":
            return (
                self._extract_vq_decoder_weights(state_dict),
                "vq_decoder.safetensors",
            )
        elif component == "audio_encoder":
            return (
                self._extract_audio_encoder_weights(state_dict),
                "audio_encoder.safetensors",
            )
        else:
            raise ValueError(f"无效的组件名: {component!r}")

    def _resolve_target_key(self, src_key: str, component: str) -> str | None:
        """解析单个源 key 在指定组件中的目标 key。

        用于 :meth:`dry_run` 中逐 key 展示映射关系。

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
        if component == "text_frontend":
            # 统一词表嵌入（恒等映射）
            if src_key == self._EMBED_KEY:
                return src_key
            return None

        elif component == "acoustic_model":
            # 静态映射（恒等）
            if src_key in self.FISH_TO_LLAMA_MAP:
                return self.FISH_TO_LLAMA_MAP[src_key]
            # 带层号的通配映射（恒等）
            match = self._LLAMA_LAYER_RE.match(src_key)
            if match:
                return f"model.layers.{match.group(1)}.{match.group(2)}"
            return None

        elif component == "vocoder":
            if ("generator" in src_key or "hifi" in src_key) and (
                "discriminator" not in src_key
            ):
                return src_key
            return None

        elif component == "vq_decoder":
            if "vq" in src_key or "decoder" in src_key:
                return src_key
            return None

        elif component == "audio_encoder":
            if "encoder" in src_key or "dac" in src_key:
                return src_key
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

        - ``text_vocab_size``：从 ``model.embed_tokens.weight`` 的 shape[0]；
        - ``hidden_size``：从 ``model.embed_tokens.weight`` 的 shape[1]；
        - ``num_layers``：LLaMA 层数；
        - ``audio_vocab_size``：从 codebook 权重推断；若未找到则回退为
          ``text_vocab_size``（统一词表）。

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
            "model_type": "fish_speech",
            "version": "0.1",
            "source": source_path,
            "conversion_date": datetime.now().isoformat(),
            "components": converted_components,
        }

        # text_vocab_size & hidden_size：从 embed_tokens 推断
        if self._EMBED_KEY in state_dict:
            embed = state_dict[self._EMBED_KEY]
            if hasattr(embed, "shape"):
                config["text_vocab_size"] = int(embed.shape[0])
                config["hidden_size"] = int(embed.shape[1])

        # audio_vocab_size：优先从 codebook 权重推断
        audio_vocab_size: int | None = None
        for key, value in state_dict.items():
            if self._CODEBOOK_RE.search(key) and hasattr(value, "shape"):
                audio_vocab_size = int(value.shape[0])
                break
        if audio_vocab_size is not None:
            config["audio_vocab_size"] = audio_vocab_size
        elif "text_vocab_size" in config:
            # 回退：统一词表，audio 与 text 共享嵌入空间
            config["audio_vocab_size"] = config["text_vocab_size"]

        # num_layers：从 LLaMA 层推断
        layer_indices: set[int] = set()
        for key in state_dict:
            match = self._LLAMA_LAYER_RE.match(key)
            if match:
                layer_indices.add(int(match.group(1)))
        if layer_indices:
            config["num_layers"] = len(layer_indices)

        return config
