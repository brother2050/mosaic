# mosaic/nodes/audio/tts_backends/weights/sovits_convert.py
"""GPT-SoVITS 权重转换器。

将 GPT-SoVITS 原始 checkpoint 转换为 Mosaic 标准权重格式
（safetensors + ``config.json``）。

GPT-SoVITS 模型结构
--------------------
GPT-SoVITS 包含以下组件：

- **text_frontend**：GPT 的文本音素嵌入层（``t2s_model.embedding.weight``）
- **acoustic_model**：GPT 自回归模型，映射为 ``GPT2LMHeadModel`` 格式
- **vocoder**：SoVITS 解码器，包含语义编码器、先验编码器、Normalizing Flow、
  条件 HiFi-GAN 解码器（``enc_p.*``、``flow.*``、``dec.*``）
- **ssl_encoder**：SSL 音频编码器（``ssl.*``），用于提取语义 token

权重映射说明
------------
GPT-SoVITS 的 GPT 部分使用自定义的 GPT-2 变体，权重命名与标准
``GPT2LMHeadModel`` 不同。:attr:`SOVITS_GPT_TO_GPT2_MAP` 定义了映射关系：

* ``t2s_model.embedding.weight`` → ``transformer.wte.weight``
* ``t2s_model.pos_embedding`` → ``transformer.wpe.weight``
* ``t2s_model.layers.{i}.attn.c_attn.*`` → ``transformer.h.{i}.attn.c_attn.*``
* ``t2s_model.layers.{i}.ln_1.*`` → ``transformer.h.{i}.ln_1.*``
* ``t2s_model.layers.{i}.mlp.c_fc.*`` → ``transformer.h.{i}.mlp.c_fc.*``
* ``t2s_model.layers.{i}.ln_2.*`` → ``transformer.h.{i}.ln_2.*``
* ``t2s_model.head.weight`` → ``lm_head.weight``

vocoder 和 ssl_encoder 组件的权重保持原名不变。

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

__all__ = ["SoVITSWeightConverter"]


class SoVITSWeightConverter(WeightConverter):
    """GPT-SoVITS 权重转换器。

    将 GPT-SoVITS 原始 checkpoint 转换为 Mosaic 标准权重格式。

    GPT-SoVITS 包含四个组件：

    - ``text_frontend``：GPT 文本音素嵌入层
    - ``acoustic_model``：GPT 自回归模型（映射为 ``GPT2LMHeadModel`` 格式）
    - ``vocoder``：SoVITS 解码器（语义编码器 + 先验编码器 + Flow + HiFi-GAN）
    - ``ssl_encoder``：SSL 音频编码器

    与基类 :class:`WeightConverter` 相比，GPT-SoVITS 额外包含
    ``ssl_encoder`` 组件。

    Examples
    --------
    >>> converter = SoVITSWeightConverter()
    >>> # 转换全部组件
    >>> result = converter.convert("/path/to/sovits.pth", "/output/dir")
    >>> # 仅转换部分组件
    >>> result = converter.convert(
    ...     "/path/to/sovits.pth", "/output/dir",
    ...     components=["acoustic_model", "vocoder"],
    ... )
    >>> # 预览映射关系（不实际转换）
    >>> mapping = converter.dry_run("/path/to/sovits.pth")
    """

    # GPT-SoVITS 组件列表（在基类基础上增加 ssl_encoder）
    COMPONENTS = (
        "text_frontend",
        "acoustic_model",
        "vocoder",
        "ssl_encoder",
    )

    # GPT-SoVITS GPT 权重 → GPT2LMHeadModel 权重映射
    # 静态映射（不含层号）；带层号的通配映射在 _map_gpt_weights 中处理
    SOVITS_GPT_TO_GPT2_MAP: dict[str, str] = {
        "t2s_model.embedding.weight": "transformer.wte.weight",
        "t2s_model.pos_embedding": "transformer.wpe.weight",
        "t2s_model.head.weight": "lm_head.weight",
        "t2s_model.final_norm.weight": "transformer.ln_f.weight",
        "t2s_model.final_norm.bias": "transformer.ln_f.bias",
        # 每一层 Transformer（带层号的通配映射，在 _map_gpt_weights 中处理）：
        # t2s_model.layers.{i}.attn.c_attn.weight    → transformer.h.{i}.attn.c_attn.weight
        # t2s_model.layers.{i}.attn.c_attn.bias      → transformer.h.{i}.attn.c_attn.bias
        # t2s_model.layers.{i}.attn.c_proj.weight    → transformer.h.{i}.attn.c_proj.weight
        # t2s_model.layers.{i}.attn.c_proj.bias      → transformer.h.{i}.attn.c_proj.bias
        # t2s_model.layers.{i}.ln_1.weight           → transformer.h.{i}.ln_1.weight
        # t2s_model.layers.{i}.ln_1.bias             → transformer.h.{i}.ln_1.bias
        # t2s_model.layers.{i}.mlp.c_fc.weight       → transformer.h.{i}.mlp.c_fc.weight
        # t2s_model.layers.{i}.mlp.c_fc.bias         → transformer.h.{i}.mlp.c_fc.bias
        # t2s_model.layers.{i}.mlp.c_proj.weight     → transformer.h.{i}.mlp.c_proj.weight
        # t2s_model.layers.{i}.mlp.c_proj.bias       → transformer.h.{i}.mlp.c_proj.bias
        # t2s_model.layers.{i}.ln_2.weight           → transformer.h.{i}.ln_2.weight
        # t2s_model.layers.{i}.ln_2.bias             → transformer.h.{i}.ln_2.bias
    }

    # GPT 层号匹配正则：t2s_model.layers.{i}.{suffix}
    _GPT_LAYER_RE = re.compile(r"^t2s_model\.layers\.(\d+)\.(.+)$")
    # GPT 层内子模块映射：{suffix_pattern: target_suffix}
    _GPT_LAYER_SUFFIX_MAP: dict[str, str] = {
        "attn.c_attn.weight": "attn.c_attn.weight",
        "attn.c_attn.bias": "attn.c_attn.bias",
        "attn.c_proj.weight": "attn.c_proj.weight",
        "attn.c_proj.bias": "attn.c_proj.bias",
        "ln_1.weight": "ln_1.weight",
        "ln_1.bias": "ln_1.bias",
        "mlp.c_fc.weight": "mlp.c_fc.weight",
        "mlp.c_fc.bias": "mlp.c_fc.bias",
        "mlp.c_proj.weight": "mlp.c_proj.weight",
        "mlp.c_proj.bias": "mlp.c_proj.bias",
        "ln_2.weight": "ln_2.weight",
        "ln_2.bias": "ln_2.bias",
    }

    # 文本嵌入 key
    _TEXT_EMBED_KEY = "t2s_model.embedding.weight"
    # SoVITS vocoder 权重前缀
    _VOCODER_PREFIXES = ("enc_p.", "flow.", "dec.")
    # SoVITS 后验编码器前缀（推理时不需要）
    _POSTERIOR_PREFIXES = ("enc_q.",)
    # SSL 编码器前缀
    _SSL_PREFIXES = ("ssl.",)

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def convert(
        self,
        source_path: str,
        output_path: str,
        components: list[str] | None = None,
    ) -> dict[str, str]:
        """将 GPT-SoVITS 原始 checkpoint 转换为 Mosaic 标准格式。

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
            "ssl_encoder"]``。

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
        if config.get("model_type") != "gpt_sovits":
            return False

        # 3. 检查各组件的 safetensors 文件是否存在并加载
        component_filenames = {
            "text_frontend": "text_frontend.safetensors",
            "acoustic_model": "acoustic_model.safetensors",
            "vocoder": "vocoder.safetensors",
            "ssl_encoder": "ssl_encoder.safetensors",
        }

        all_weights: dict[str, Any] = {}
        for comp, filename in component_filenames.items():
            fpath = os.path.join(converted_path, filename)
            if not os.path.isfile(fpath):
                continue
            try:
                from safetensors.torch import load_file  # type: ignore

                all_weights.update(load_file(fpath))
            except ImportError:
                return False
            except Exception:
                return False

        if not all_weights:
            return False

        # 4. 检查 key 和 shape 与 config.json 一致

        # 检查 text_vocab_size & hidden_size：从 wte 推断
        embed_key = "transformer.wte.weight"
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
            if (m := re.match(r"^transformer\.h\.(\d+)\.", k))
        }
        if layer_keys and "num_layers" in config:
            if len(layer_keys) != config["num_layers"]:
                return False

        # 检查 ssl_vocab_size：从 semantic_encoder embedding 推断
        ssl_embed_keys = [
            k for k in all_weights
            if "semantic_encoder" in k and "embedding" in k
        ]
        if ssl_embed_keys and "ssl_vocab_size" in config:
            first_embed = all_weights[ssl_embed_keys[0]]
            if hasattr(first_embed, "shape"):
                if first_embed.shape[0] != config["ssl_vocab_size"]:
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
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"源路径不存在: {source_path}")

        state_dict = self._load_checkpoint(source_path)

        if components is None:
            components = list(self.COMPONENTS)
        else:
            for comp in components:
                if comp not in self.COMPONENTS:
                    raise ValueError(
                        f"无效的组件名: {comp!r}，"
                        f"可选组件: {list(self.COMPONENTS)}"
                    )

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

                shape = tuple(value.shape) if hasattr(value, "shape") else None
                print(f"  {src_key}  ->  {tgt_key}    shape={shape}")
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
        或 ``"weight"`` 键，则自动解包。

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
            """解包 checkpoint 字典。"""
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt and isinstance(
                    ckpt["state_dict"], dict
                ):
                    return ckpt["state_dict"]
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    return ckpt["model"]
                if "weight" in ckpt and isinstance(ckpt, dict):
                    return ckpt
            if isinstance(ckpt, dict):
                return ckpt
            raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")

        if os.path.isdir(source_path):
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

    def _map_gpt_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """按 :attr:`SOVITS_GPT_TO_GPT2_MAP` 映射 GPT 权重为 ``GPT2LMHeadModel`` 格式。

        处理两类映射：

        1. **静态映射**：``t2s_model.embedding.weight``、``t2s_model.head.weight``
           等显式列出的 key；
        2. **带层号的通配映射**：``t2s_model.layers.{i}.xxx`` →
           ``transformer.h.{i}.xxx``。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            映射后的权重字典（key 为 GPT2LMHeadModel 格式）。
        """
        mapped: dict[str, Any] = {}

        for key, value in state_dict.items():
            # 1. 静态映射
            if key in self.SOVITS_GPT_TO_GPT2_MAP:
                mapped[self.SOVITS_GPT_TO_GPT2_MAP[key]] = value
                continue

            # 2. 带层号的通配映射：t2s_model.layers.{i}.{suffix}
            match = self._GPT_LAYER_RE.match(key)
            if match:
                layer_idx = match.group(1)
                suffix = match.group(2)
                if suffix in self._GPT_LAYER_SUFFIX_MAP:
                    mapped[
                        f"transformer.h.{layer_idx}.{self._GPT_LAYER_SUFFIX_MAP[suffix]}"
                    ] = value

        return mapped

    def _extract_text_frontend_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取文本音素嵌入权重。

        提取 ``t2s_model.embedding.weight``（GPT 的文本音素嵌入）。
        映射为 ``transformer.wte.weight``。

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
        if self._TEXT_EMBED_KEY in state_dict:
            mapped["transformer.wte.weight"] = state_dict[self._TEXT_EMBED_KEY]
        return mapped

    def _extract_vocoder_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取 SoVITS 解码器权重。

        提取 key 以 ``enc_p.``、``flow.``、``dec.`` 开头的权重，
        并过滤掉后验编码器权重（``enc_q.``）。

        Vocoder 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            SoVITS 解码器权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            # 跳过后验编码器
            if any(key.startswith(p) for p in self._POSTERIOR_PREFIXES):
                continue
            # 匹配 vocoder 前缀
            if any(key.startswith(p) for p in self._VOCODER_PREFIXES):
                extracted[key] = value
        return extracted

    def _extract_ssl_encoder_weights(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """提取 SSL 音频编码器权重。

        提取 key 以 ``ssl.`` 开头的权重。

        SSL Encoder 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            SSL 编码器权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            if any(key.startswith(p) for p in self._SSL_PREFIXES):
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
                self._extract_text_frontend_weights(state_dict),
                "text_frontend.safetensors",
            )
        elif component == "acoustic_model":
            return (
                self._map_gpt_weights(state_dict),
                "acoustic_model.safetensors",
            )
        elif component == "vocoder":
            return (
                self._extract_vocoder_weights(state_dict),
                "vocoder.safetensors",
            )
        elif component == "ssl_encoder":
            return (
                self._extract_ssl_encoder_weights(state_dict),
                "ssl_encoder.safetensors",
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
            if src_key == self._TEXT_EMBED_KEY:
                return "transformer.wte.weight"
            return None

        elif component == "acoustic_model":
            # 静态映射
            if src_key in self.SOVITS_GPT_TO_GPT2_MAP:
                return self.SOVITS_GPT_TO_GPT2_MAP[src_key]
            # 带层号的通配映射
            match = self._GPT_LAYER_RE.match(src_key)
            if match:
                layer_idx = match.group(1)
                suffix = match.group(2)
                if suffix in self._GPT_LAYER_SUFFIX_MAP:
                    return (
                        f"transformer.h.{layer_idx}."
                        f"{self._GPT_LAYER_SUFFIX_MAP[suffix]}"
                    )
            return None

        elif component == "vocoder":
            # 跳过后验编码器
            if any(src_key.startswith(p) for p in self._POSTERIOR_PREFIXES):
                return None
            if any(src_key.startswith(p) for p in self._VOCODER_PREFIXES):
                return src_key
            return None

        elif component == "ssl_encoder":
            if any(src_key.startswith(p) for p in self._SSL_PREFIXES):
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

        - ``text_vocab_size``：从 ``t2s_model.embedding.weight`` 的 shape[0]；
        - ``hidden_size``：从 ``t2s_model.embedding.weight`` 的 shape[1]；
        - ``num_layers``：GPT 层数；
        - ``ssl_vocab_size``：从 ``enc_p.embedding.weight`` 推断（如存在）。

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
            "model_type": "gpt_sovits",
            "version": "0.1",
            "source": source_path,
            "conversion_date": datetime.now().isoformat(),
            "components": converted_components,
        }

        # text_vocab_size & hidden_size：从 embedding 推断
        if self._TEXT_EMBED_KEY in state_dict:
            embed = state_dict[self._TEXT_EMBED_KEY]
            if hasattr(embed, "shape"):
                config["text_vocab_size"] = int(embed.shape[0])
                config["hidden_size"] = int(embed.shape[1])

        # num_layers：从 GPT 层推断
        layer_indices: set[int] = set()
        for key in state_dict:
            match = self._GPT_LAYER_RE.match(key)
            if match:
                layer_indices.add(int(match.group(1)))
        if layer_indices:
            config["num_layers"] = len(layer_indices)

        # ssl_vocab_size：从 enc_p.embedding 推断
        ssl_embed_key = "enc_p.embedding.weight"
        if ssl_embed_key in state_dict:
            ssl_embed = state_dict[ssl_embed_key]
            if hasattr(ssl_embed, "shape"):
                config["ssl_vocab_size"] = int(ssl_embed.shape[0])
        else:
            # 回退：默认 SSL 词表大小
            config["ssl_vocab_size"] = 768

        return config
