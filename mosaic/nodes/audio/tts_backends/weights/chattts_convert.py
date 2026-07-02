"""ChatTTS 权重转换器。

将 ChatTTS 原始 checkpoint 转换为 Mosaic 标准权重格式
（safetensors + ``config.json``）。

ChatTTS 模型结构
----------------
ChatTTS 包含以下组件：

- **text_frontend**：文本/音频 token 嵌入层（``emb_text``、``emb_code``）
- **acoustic_model**：GPT 自回归模型，映射为 ``LlamaForCausalLM`` 格式
- **vocoder**：Vocos 声码器，将 mel 频谱转换为波形
- **dvae**：离散 VAE，用于音频 token 量化

与基类 :class:`WeightConverter` 相比，ChatTTS 额外包含 DVAE 组件。

权重映射说明
------------
ChatTTS 的 GPT 模型权重以 ``gpt.`` 为前缀，需要映射为
``LlamaForCausalLM`` 的标准命名：

- ``gpt.model.embed_tokens.weight`` → ``model.embed_tokens.weight``
- ``gpt.model.norm.weight`` → ``model.norm.weight``
- ``gpt.lm_head.weight`` → ``lm_head.weight``
- ``gpt.model.layers.{i}.*`` → ``model.layers.{i}.*``（逐层通配）

嵌入层权重（``emb_text``、``emb_code``）属于 text_frontend 组件，
仅做前缀裁剪。Vocos 与 DVAE 权重保持原名不变。

依赖说明
--------
``torch`` 与 ``safetensors`` 均为惰性导入，仅在加载/保存权重时需要；
模块导入本身不依赖它们。
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any

from mosaic.nodes.audio.tts_backends.weights.converter import WeightConverter

logger = logging.getLogger(__name__)

__all__ = ["ChatTTSWeightConverter"]


class ChatTTSWeightConverter(WeightConverter):
    """ChatTTS 权重转换器。

    将 ChatTTS 原始 checkpoint 转换为 Mosaic 标准权重格式。

    ChatTTS 包含四个组件：

    - ``text_frontend``：文本/音频 token 嵌入层（``emb_text``、``emb_code``）
    - ``acoustic_model``：GPT 自回归模型（映射为 ``LlamaForCausalLM`` 格式）
    - ``vocoder``：Vocos 声码器
    - ``dvae``：离散 VAE

    与基类 :class:`WeightConverter` 相比，ChatTTS 额外包含 DVAE 组件，
    因此 ``COMPONENTS`` 中增加了 ``"dvae"``。

    Examples
    --------
    >>> converter = ChatTTSWeightConverter()
    >>> # 转换全部组件
    >>> result = converter.convert("/path/to/chattts.pt", "/output/dir")
    >>> # 仅转换部分组件
    >>> result = converter.convert(
    ...     "/path/to/chattts.pt", "/output/dir",
    ...     components=["acoustic_model", "vocoder"],
    ... )
    >>> # 预览映射关系（不实际转换）
    >>> mapping = converter.dry_run("/path/to/chattts.pt")
    """

    # ChatTTS 组件列表（在基类基础上增加 dvae）
    COMPONENTS = ("text_frontend", "acoustic_model", "vocoder", "dvae")

    # ChatTTS GPT 权重 → LlamaForCausalLM 权重映射
    # 静态映射（不含层号）；带层号的通配映射在 _map_gpt_weights 中处理
    GPT_TO_LLAMA_MAP: dict[str, str] = {
        "gpt.model.embed_tokens.weight": "model.embed_tokens.weight",
        "gpt.model.norm.weight": "model.norm.weight",
        "gpt.lm_head.weight": "lm_head.weight",
        # 每一层 Transformer（带层号的通配映射，在 _map_gpt_weights 中处理）：
        # gpt.model.layers.{i}.self_attn.q_proj.weight          → model.layers.{i}.self_attn.q_proj.weight
        # gpt.model.layers.{i}.self_attn.k_proj.weight          → model.layers.{i}.self_attn.k_proj.weight
        # gpt.model.layers.{i}.self_attn.v_proj.weight          → model.layers.{i}.self_attn.v_proj.weight
        # gpt.model.layers.{i}.self_attn.o_proj.weight          → model.layers.{i}.self_attn.o_proj.weight
        # gpt.model.layers.{i}.mlp.gate_proj.weight             → model.layers.{i}.mlp.gate_proj.weight
        # gpt.model.layers.{i}.mlp.down_proj.weight             → model.layers.{i}.mlp.down_proj.weight
        # gpt.model.layers.{i}.mlp.up_proj.weight               → model.layers.{i}.mlp.up_proj.weight
        # gpt.model.layers.{i}.input_layernorm.weight           → model.layers.{i}.input_layernorm.weight
        # gpt.model.layers.{i}.post_attention_layernorm.weight  → model.layers.{i}.post_attention_layernorm.weight
    }

    # Embed 层权重映射（静态部分；emb_code 的带编号通配在 _map_embed_weights 中处理）
    EMBED_KEY_MAP: dict[str, str] = {
        "gpt.emb_text.weight": "emb_text.weight",
        # 带编号的通配映射（在 _map_embed_weights 中处理）：
        # gpt.emb_code.{i}.weight → emb_code.{i}.weight
    }

    # 内部前缀常量，避免魔法字符串
    _GPT_MODEL_PREFIX = "gpt.model."
    _LLAMA_MODEL_PREFIX = "model."

    # GPT 层号匹配正则：gpt.model.layers.{i}.{suffix}
    _GPT_LAYER_RE = re.compile(r"^gpt\.model\.layers\.(\d+)\.(.+)$")
    # emb_code 编号匹配正则：gpt.emb_code.{i}.weight
    _EMB_CODE_RE = re.compile(r"^gpt\.emb_code\.(\d+)\.weight$")

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def convert(
        self,
        source_path: str,
        output_path: str,
        components: list[str] | None = None,
    ) -> dict[str, str]:
        """将 ChatTTS 原始 checkpoint 转换为 Mosaic 标准格式。

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
            可选值：``["text_frontend", "acoustic_model", "vocoder", "dvae"]``。

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
        if config.get("model_type") != "chattts":
            return False

        # 3. 检查各组件的 safetensors 文件是否存在并加载
        component_filenames = {
            "text_frontend": "text_frontend.safetensors",
            "acoustic_model": "acoustic_model.safetensors",
            "vocoder": "vocoder.safetensors",
            "dvae": "dvae.safetensors",
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

        if not all_weights:
            return False

        # 4. 检查 key 和 shape 与 config.json 一致

        # 检查 num_text_tokens
        if "emb_text.weight" in all_weights and "num_text_tokens" in config:
            if all_weights["emb_text.weight"].shape[0] != config["num_text_tokens"]:
                return False

        # 检查 hidden_size
        if "emb_text.weight" in all_weights and "hidden_size" in config:
            if all_weights["emb_text.weight"].shape[1] != config["hidden_size"]:
                return False

        # 检查 num_audio_tokens 和 num_vq
        # 转换后的 key 为 emb_code.{i}.weight（无 gpt. 前缀）
        code_keys = [
            k for k in all_weights if re.match(r"^emb_code\.\d+\.weight$", k)
        ]
        if code_keys:
            if "num_audio_tokens" in config:
                for k in code_keys:
                    if all_weights[k].shape[0] != config["num_audio_tokens"]:
                        return False
            if "num_vq" in config:
                if len(code_keys) != config["num_vq"]:
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
            logger.info("\n%s\n组件: %s\n%s", "=" * 70, comp, "=" * 70)

            comp_mapping: dict[str, str] = {}
            for src_key, value in state_dict.items():
                tgt_key = self._resolve_target_key(src_key, comp)
                if tgt_key is None:
                    continue

                # 获取 shape
                shape = tuple(value.shape) if hasattr(value, "shape") else None
                logger.info("  %s  →  %s    shape=%s", src_key, tgt_key, shape)
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

    def _map_gpt_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """按 ``GPT_TO_LLAMA_MAP`` 映射 GPT 权重为 ``LlamaForCausalLM`` 格式。

        处理两类映射：

        1. **静态映射**：``gpt.model.embed_tokens.weight`` 等显式列出的 key；
        2. **带层号的通配映射**：``gpt.model.layers.{i}.xxx`` →
           ``model.layers.{i}.xxx``；
        3. **其他 gpt.model.* 权重**：裁剪 ``gpt.`` 前缀。

        仅提取 ``gpt.model.*`` 和 ``gpt.lm_head.*`` 的权重，不包含
        ``gpt.emb_text`` / ``gpt.emb_code``（属于 text_frontend）。

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
            if key in self.GPT_TO_LLAMA_MAP:
                mapped[self.GPT_TO_LLAMA_MAP[key]] = value
                continue

            # 2. 带层号的通配映射：gpt.model.layers.{i}.{suffix}
            match = self._GPT_LAYER_RE.match(key)
            if match:
                layer_idx, suffix = match.group(1), match.group(2)
                mapped[f"model.layers.{layer_idx}.{suffix}"] = value
                continue

            # 3. 其他 gpt.model.* 权重：裁剪 gpt. 前缀
            if key.startswith(self._GPT_MODEL_PREFIX):
                new_key = self._LLAMA_MODEL_PREFIX + key[len(
                    self._GPT_MODEL_PREFIX
                ):]
                mapped[new_key] = value

        return mapped

    def _map_embed_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """提取 ``emb_text`` 和 ``emb_code`` 权重并映射 key 名。

        映射规则：

        - ``gpt.emb_text.weight`` → ``emb_text.weight``（静态映射）；
        - ``gpt.emb_code.{i}.weight`` → ``emb_code.{i}.weight``（通配映射）。

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

        for key, value in state_dict.items():
            # 1. emb_text 静态映射
            if key in self.EMBED_KEY_MAP:
                mapped[self.EMBED_KEY_MAP[key]] = value
                continue

            # 2. emb_code.{i} 通配映射
            match = self._EMB_CODE_RE.match(key)
            if match:
                idx = match.group(1)
                mapped[f"emb_code.{idx}.weight"] = value

        return mapped

    def _extract_vocoder_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """提取 Vocos 声码器权重。

        提取 key 中包含 ``"vocos"`` 或 ``"decoder"`` 的权重。
        为避免与 DVAE 的 ``dvae.decoder`` 权重冲突，包含 ``"dvae"`` 的
        key 不会被提取到 vocoder 中。

        Vocos 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            Vocos 声码器权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            # 包含 "vocos" 或 "decoder"，但排除 "dvae"（避免与 DVAE 冲突）
            if ("vocos" in key or "decoder" in key) and "dvae" not in key:
                extracted[key] = value
        return extracted

    def _extract_dvae_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """提取 DVAE 权重。

        提取 key 中包含 ``"dvae"`` 或 ``"encoder"`` 的权重。
        为避免与 Vocos 权重冲突，包含 ``"vocos"`` 的 key 不会被提取
        到 DVAE 中。

        DVAE 权重保持原名不变。

        Parameters
        ----------
        state_dict : dict[str, Any]
            原始权重字典。

        Returns
        -------
        dict[str, Any]
            DVAE 权重字典。
        """
        extracted: dict[str, Any] = {}
        for key, value in state_dict.items():
            # 包含 "dvae" 或 "encoder"，但排除 "vocos"（避免冲突）
            if ("dvae" in key or "encoder" in key) and "vocos" not in key:
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
            return self._map_embed_weights(state_dict), "text_frontend.safetensors"
        elif component == "acoustic_model":
            return self._map_gpt_weights(state_dict), "acoustic_model.safetensors"
        elif component == "vocoder":
            return self._extract_vocoder_weights(state_dict), "vocoder.safetensors"
        elif component == "dvae":
            return self._extract_dvae_weights(state_dict), "dvae.safetensors"
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
            # emb_text 静态映射
            if src_key in self.EMBED_KEY_MAP:
                return self.EMBED_KEY_MAP[src_key]
            # emb_code.{i} 通配映射
            match = self._EMB_CODE_RE.match(src_key)
            if match:
                return f"emb_code.{match.group(1)}.weight"
            return None

        elif component == "acoustic_model":
            # 静态映射
            if src_key in self.GPT_TO_LLAMA_MAP:
                return self.GPT_TO_LLAMA_MAP[src_key]
            # 带层号的通配映射
            match = self._GPT_LAYER_RE.match(src_key)
            if match:
                return f"model.layers.{match.group(1)}.{match.group(2)}"
            # 其他 gpt.model.* 权重
            if src_key.startswith(self._GPT_MODEL_PREFIX):
                return self._LLAMA_MODEL_PREFIX + src_key[len(
                    self._GPT_MODEL_PREFIX
                ):]
            return None

        elif component == "vocoder":
            if ("vocos" in src_key or "decoder" in src_key) and "dvae" not in src_key:
                return src_key
            return None

        elif component == "dvae":
            if ("dvae" in src_key or "encoder" in src_key) and "vocos" not in src_key:
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

        - ``num_text_tokens``：从 ``emb_text.weight`` 的 shape[0]；
        - ``num_audio_tokens``：从 ``emb_code.0.weight`` 的 shape[0]；
        - ``num_vq``：``emb_code`` 的数量；
        - ``hidden_size``：从 ``emb_text.weight`` 或
          ``model.embed_tokens.weight`` 的 shape[1]；
        - ``num_layers``：GPT 层数。

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
            "model_type": "chattts",
            "version": "0.1",
            "source": source_path,
            "conversion_date": datetime.now().isoformat(),
            "components": converted_components,
        }

        # num_text_tokens & hidden_size：从 emb_text 推断
        emb_text_key = "gpt.emb_text.weight"
        if emb_text_key in state_dict:
            emb_text = state_dict[emb_text_key]
            if hasattr(emb_text, "shape"):
                config["num_text_tokens"] = int(emb_text.shape[0])
                config["hidden_size"] = int(emb_text.shape[1])

        # num_vq & num_audio_tokens：从 emb_code 推断
        code_indices: list[int] = []
        for key in state_dict:
            match = self._EMB_CODE_RE.match(key)
            if match:
                code_indices.append(int(match.group(1)))
        if code_indices:
            config["num_vq"] = len(code_indices)
            first_code_key = f"gpt.emb_code.{min(code_indices)}.weight"
            first_code = state_dict[first_code_key]
            if hasattr(first_code, "shape"):
                config["num_audio_tokens"] = int(first_code.shape[0])

        # hidden_size 回退：从 embed_tokens 推断
        if "hidden_size" not in config:
            embed_key = "gpt.model.embed_tokens.weight"
            if embed_key in state_dict and hasattr(
                state_dict[embed_key], "shape"
            ):
                config["hidden_size"] = int(state_dict[embed_key].shape[1])

        # num_layers：从 GPT 层推断
        layer_indices: set[int] = set()
        for key in state_dict:
            match = self._GPT_LAYER_RE.match(key)
            if match:
                layer_indices.add(int(match.group(1)))
        if layer_indices:
            # 使用 LlamaConfig 的标准字段名 num_hidden_layers
            # （而非 num_layers，LlamaConfig 不识别 num_layers）
            config["num_hidden_layers"] = len(layer_indices)
            config["num_layers"] = len(layer_indices)  # 向后兼容

        # 补齐 LlamaConfig 必需字段
        if "hidden_size" in config:
            hs = config["hidden_size"]
            # num_attention_heads：ChatTTS 默认 12（768/64=12）
            if "num_attention_heads" not in config:
                config["num_attention_heads"] = max(1, hs // 64)
            # num_key_value_heads：GQA=MHA
            if "num_key_value_heads" not in config:
                config["num_key_value_heads"] = config["num_attention_heads"]
            # intermediate_size：标准 4x hidden_size
            if "intermediate_size" not in config:
                config["intermediate_size"] = hs * 4
            # max_position_embeddings
            if "max_position_embeddings" not in config:
                config["max_position_embeddings"] = 2048

        return config
