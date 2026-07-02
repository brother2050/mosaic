# mosaic/nodes/audio/tts_backends/acoustic_models/llama_ar.py
"""基于 Llama 架构的自回归声学模型。

本模块提供 ChatTTS 和 Fish Speech 共用的 LLaMA 自回归声学模型框架。

Layer 2: 声学模型层。将文本 token ids 转换为离散音频码 token。

类层次
------
::

    AcousticModel (抽象基类)
      └── LlamaARModelBase (公共逻辑)
            ├── LlamaARModel       (ChatTTS 特化：双路径 Embed + 多 VQ 输出)
            └── FishLlamaARModel   (Fish Speech 特化：统一 Embed + 语音克隆)

公共逻辑
--------
* LlamaForCausalLM 的加载/卸载
* dtype 解析与设备迁移
* KV Cache 管理
* 通用的采样逻辑（temperature、top_k、top_p、repetition_penalty）
* 停止条件检查框架
* 流式生成框架

显存需求
--------
* ``float16`` 精度：约 2-4 GB GPU 显存
* ``float32`` 精度：约 4-8 GB GPU 显存

设计要点
--------
* ``torch`` / ``transformers`` / ``safetensors`` 等重依赖采用惰性导入，使本
  模块在未安装这些依赖时仍可被导入与继承。
* ``token_ids`` / ``speaker_embedding`` 等参数类型用 :data:`~typing.Any`
  标注，避免在模块顶层硬依赖 ``torch``。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel

__all__ = ["LlamaARModelBase", "LlamaARModel", "DualEmbedding"]


class LlamaARModelBase(AcousticModel):
    """基于 LlamaForCausalLM 的自回归声学模型基类。

    为 ChatTTS (:class:`LlamaARModel`) 和 Fish Speech
    (:class:`FishLlamaARModel`) 提供共用的 LLaMA 加载、卸载、采样逻辑。

    子类需要实现:
    - :meth:`load_weights`: 加载权重（含 Embed 层创建）
    - :meth:`generate`: 自回归生成
    - :meth:`generate_stream`: 流式生成

    Attributes
    ----------
    model_type : str
        模型类型，固定为 ``"ar"``。
    """

    model_type: str = "ar"

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 24,
        max_position_embeddings: int = 2048,
        use_flash_attention: bool = True,
    ) -> None:
        """初始化公共属性。

        Parameters
        ----------
        model_path : str
            模型路径。
        hidden_size : int
            隐藏层维度。
        num_heads : int
            注意力头数。
        num_layers : int
            Transformer 层数。
        max_position_embeddings : int
            最大位置编码长度。
        use_flash_attention : bool
            是否使用 Flash Attention。
        """
        self._model_path: str = model_path
        self._hidden_size: int = hidden_size
        self._num_heads: int = num_heads
        self._num_layers: int = num_layers
        self._max_position_embeddings: int = max_position_embeddings
        self._use_flash_attention: bool = use_flash_attention

        # 模型实例（load_weights 后填充）
        self._model: Any = None
        self._embed_layer: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # 公共辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_dtype(dtype: str) -> Any:
        """将 dtype 字符串解析为 torch dtype。

        Parameters
        ----------
        dtype : str
            ``"float16"`` / ``"float32"`` / ``"bfloat16"``。

        Returns
        -------
        torch.dtype

        Raises
        ------
        ValueError
            不支持的 dtype 字符串。
        """
        import torch

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Supported: {list(dtype_map.keys())}"
            )
        return dtype_map[dtype]

    def _create_llama_config(
        self, weights_path: str, torch_dtype: Any, vocab_size: int
    ) -> Any:
        """加载或创建 LlamaConfig。

        Parameters
        ----------
        weights_path : str
            权重目录路径。
        torch_dtype : torch.dtype
            目标精度。
        vocab_size : int
            词表大小。

        Returns
        -------
        LlamaConfig
        """
        from transformers import LlamaConfig  # type: ignore

        config_path = os.path.join(weights_path, "config.json")
        if os.path.exists(config_path):
            config = LlamaConfig.from_pretrained(weights_path)
            config.vocab_size = vocab_size
            # 关键：将 config.json 的维度回写到 self，确保 DualEmbedding
            # 与 LlamaForCausalLM 使用相同的 hidden_size。
            # 否则 Llama 用 config.json 的 hidden_size(如768)，
            # DualEmbedding 用构造默认的 self._hidden_size(如512)，
            # 导致 generate 时维度不匹配崩溃。
            self._hidden_size = config.hidden_size
            self._num_layers = getattr(config, "num_hidden_layers", self._num_layers)
            self._num_heads = getattr(config, "num_attention_heads", self._num_heads)
            self._max_position_embeddings = getattr(
                config, "max_position_embeddings", self._max_position_embeddings
            )
        else:
            config = LlamaConfig(
                vocab_size=vocab_size,
                hidden_size=self._hidden_size,
                num_hidden_layers=self._num_layers,
                num_attention_heads=self._num_heads,
                num_key_value_heads=self._num_heads,  # GQA=MHA，兼容 transformers v5 cache
                max_position_embeddings=self._max_position_embeddings,
                torch_dtype=torch_dtype,
            )
        return config

    def _load_llama_weights(self, model: Any, weights_path: str) -> None:
        """将权重加载到 LlamaForCausalLM 实例。

        尝试顺序：safetensors → pytorch checkpoint → from_pretrained。

        Parameters
        ----------
        model : LlamaForCausalLM
            模型实例。
        weights_path : str
            权重目录路径。
        """
        import torch

        safetensors_path = os.path.join(weights_path, "acoustic_model.safetensors")
        pytorch_path = os.path.join(weights_path, "acoustic_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file  # type: ignore

            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict, strict=False)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=False)
        elif os.path.isdir(weights_path):
            try:
                from transformers import LlamaForCausalLM  # type: ignore

                model = LlamaForCausalLM.from_pretrained(weights_path)
            except Exception:  # noqa: BLE001
                pass  # 没有可用权重，使用随机初始化

    def unload_weights(self) -> None:
        """释放模型权重。

        释放 ``_model`` 和 ``_embed_layer``，设置为 ``None``，清空 CUDA 缓存
        （如果在 CUDA 上），并设置 ``_is_loaded = False``。
        """
        self._model = None
        self._embed_layer = None

        try:
            import torch

            from mosaic.core.device_utils import empty_device_cache

            empty_device_cache()
        except ImportError:
            pass

        self._is_loaded = False

    def get_input_embeddings(self) -> Any:
        """返回输入嵌入层。

        Returns
        -------
        Any | None
            Embed 层实例；未加载权重时返回 ``None``。
        """
        return self._embed_layer

    def get_output_head(self) -> Any:
        """返回输出头。

        Returns
        -------
        Any | None
            ``LlamaForCausalLM`` 的 ``lm_head``；未加载权重时返回 ``None``。
        """
        if self._model is not None:
            return self._model.lm_head
        return None

    # ------------------------------------------------------------------
    # 采样工具（静态方法，子类和外部均可使用）
    # ------------------------------------------------------------------
    @staticmethod
    def _top_k_filtering(logits: Any, top_k: int) -> Any:
        """保留 logits 中 top_k 个最大值，其余设为 -inf。"""
        import torch

        if top_k <= 0:
            return logits

        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        indices_to_remove = logits < kth_vals
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    @staticmethod
    def _top_p_filtering(logits: Any, top_p: float) -> Any:
        """nucleus sampling：保留累积概率超过 top_p 的最小 token 集合。"""
        import torch

        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    @staticmethod
    def _apply_repetition_penalty(
        logits: Any, generated_ids: Any, penalty: float
    ) -> Any:
        """对已生成 token 的 logits 除以 penalty（降低概率）。"""
        import torch

        if penalty == 1.0:
            return logits

        for batch_idx in range(logits.size(0)):
            ids = generated_ids[batch_idx].unique().long()
            # 安全 clamp：过滤超出 logits 维度的 id，防止越界索引
            ids = ids[(ids >= 0) & (ids < logits.size(-1))]
            if ids.numel() == 0:
                continue
            batch_logits = logits[batch_idx]
            gathered = batch_logits[ids]
            gathered = torch.where(
                gathered > 0,
                gathered / penalty,
                gathered * penalty,
            )
            batch_logits[ids] = gathered
            logits[batch_idx] = batch_logits
        return logits


class DualEmbedding:
    """ChatTTS 双路径嵌入层。

    文本位置使用 ``emb_text``，音频位置使用多个 ``emb_code`` 求和。

    本类是对 :class:`torch.nn.Module` 的轻量包装，实际的 ``nn.Module`` 子类
    在首次实例化时惰性构建（需要 ``torch`` 可用）。

    Attributes
    ----------
    emb_text : torch.nn.Embedding
        文本 token 嵌入层。
    emb_code : torch.nn.ModuleList
        各 VQ 码本的音频 token 嵌入层列表。
    """

    def __new__(
        cls,
        num_text_tokens: int,
        num_audio_tokens: int,
        num_vq: int,
        hidden_size: int,
    ) -> "DualEmbedding":
        """构建真实的 ``nn.Module`` 子类实例。"""
        import torch.nn as nn

        class _DualEmbeddingImpl(nn.Module):
            """ChatTTS 双路径嵌入层的真实实现。"""

            def __init__(self) -> None:
                super().__init__()
                self.emb_text = nn.Embedding(num_text_tokens, hidden_size)
                self.emb_code = nn.ModuleList(
                    [nn.Embedding(num_audio_tokens, hidden_size) for _ in range(num_vq)]
                )

            def forward(
                self, input_ids: Any, text_mask: Any | None = None
            ) -> Any:
                """前向计算。"""
                import torch

                if text_mask is not None:
                    # 文本位置：clamp 到 [0, num_text_tokens-1]，防止越界
                    text_ids = input_ids.long().clamp(min=0, max=num_text_tokens - 1)
                    text_emb = self.emb_text(text_ids)
                    audio_emb = torch.zeros_like(text_emb)
                    for i in range(num_vq):
                        # 音频位置：减偏移后 clamp 到 [0, num_audio_tokens-1]
                        audio_ids = (
                            input_ids[..., i].long() - num_text_tokens - i * num_audio_tokens
                        ).clamp(min=0, max=num_audio_tokens - 1)
                        audio_emb = audio_emb + self.emb_code[i](audio_ids)
                    emb = torch.where(text_mask.bool().unsqueeze(-1), text_emb, audio_emb)
                    return emb

                if input_ids.dim() == 3 and input_ids.size(-1) == num_vq:
                    emb = torch.zeros(
                        *input_ids.shape[:-1],
                        hidden_size,
                        device=input_ids.device,
                        dtype=self.emb_text.weight.dtype,
                    )
                    for i in range(num_vq):
                        # 减去文本 token 偏移量，使音频 token ID 落在 emb_code[i] 的合法范围
                        audio_ids = input_ids[..., i].long() - num_text_tokens - i * num_audio_tokens
                        audio_ids = audio_ids.clamp(min=0, max=num_audio_tokens - 1)
                        emb = emb + self.emb_code[i](audio_ids)
                    return emb

                # 2D 路径：纯文本 token，clamp 防止字符级回退产生的越界 id
                safe_ids = input_ids.long().clamp(min=0, max=num_text_tokens - 1)
                return self.emb_text(safe_ids)

        impl = _DualEmbeddingImpl()
        impl.__class__ = type(
            "DualEmbedding", (_DualEmbeddingImpl,), {"__doc__": cls.__doc__}
        )
        return impl


class LlamaARModel(LlamaARModelBase):
    """ChatTTS 基于 Llama 的自回归声学模型。

    使用 :class:`transformers.LlamaForCausalLM` 作为骨干网络，配合
    :class:`DualEmbedding` 双路径嵌入层，将文本 token ids 自回归地转换为
    多组 VQ 音频码 token。

    Attributes
    ----------
    model_type : str
        模型类型，固定为 ``"ar"``。
    vocab_size : int
        词表大小，等于 ``num_text_tokens + num_audio_tokens * num_vq``。
    hidden_size : int
        隐藏层维度。
    """

    vocab_size: int = 0
    hidden_size: int = 0

    def __init__(
        self,
        model_path: str,
        num_vq: int = 4,
        num_audio_tokens: int = 1024,
        num_text_tokens: int = 0,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 24,
        max_position_embeddings: int = 2048,
        use_flash_attention: bool = True,
    ) -> None:
        """初始化 LlamaARModel。

        Parameters
        ----------
        model_path : str
            模型路径（包含 ``config.json`` 与权重文件）。
        num_vq : int
            VQ 码本组数，默认 ``4``。
        num_audio_tokens : int
            每个码本的 token 数，默认 ``1024``。
        num_text_tokens : int
            文本词表大小。
        hidden_size : int
            隐藏层维度，默认 ``512``。
        num_heads : int
            注意力头数，默认 ``8``。
        num_layers : int
            Transformer 层数，默认 ``24``。
        max_position_embeddings : int
            最大位置编码长度，默认 ``2048``。
        use_flash_attention : bool
            是否使用 Flash Attention 加速，默认 ``True``。
        """
        super().__init__(
            model_path=model_path,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            use_flash_attention=use_flash_attention,
        )

        # ChatTTS 特有属性
        self.vocab_size = num_text_tokens + num_audio_tokens * num_vq
        self.hidden_size = hidden_size
        self._num_vq: int = num_vq
        self._num_audio_tokens: int = num_audio_tokens
        self._num_text_tokens: int = num_text_tokens

    # ------------------------------------------------------------------
    # 权重加载
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载模型权重。

        加载步骤：

        1. 解析 dtype 字符串为 torch dtype；
        2. 从 ``weights_path/config.json`` 加载 :class:`LlamaConfig`（或用
           构造参数创建）；
        3. 创建 :class:`LlamaForCausalLM` 实例；
        4. 加载权重（从 safetensors 或 pytorch checkpoint）；
        5. 创建 :class:`DualEmbedding` 实例；
        6. 加载 Embed 权重（如果有单独的 embed 权重文件）；
        7. 移动到指定 device 和 dtype；
        8. 设置为 eval 模式；
        9. 设置 ``_is_loaded = True``。
        """
        import torch

        # 1. 解析 dtype
        torch_dtype = self._parse_dtype(dtype)

        # 2-3. 创建 LlamaConfig 和 LlamaForCausalLM
        from transformers import LlamaForCausalLM  # type: ignore

        config = self._create_llama_config(weights_path, torch_dtype, self.vocab_size)
        model = LlamaForCausalLM(config)

        # 4. 加载权重
        self._load_llama_weights(model, weights_path)

        # 5. 创建 DualEmbedding 实例
        embed_layer = DualEmbedding(
            num_text_tokens=self._num_text_tokens,
            num_audio_tokens=self._num_audio_tokens,
            num_vq=self._num_vq,
            hidden_size=self._hidden_size,
        )

        # 6. 加载 Embed 权重
        embed_path = os.path.join(weights_path, "embed.safetensors")
        if os.path.exists(embed_path):
            from safetensors.torch import load_file  # type: ignore

            embed_state = load_file(embed_path)
            embed_layer.load_state_dict(embed_state, strict=False)

        # 7. 移动到指定 device 和 dtype
        model = model.to(device=device, dtype=torch_dtype)
        embed_layer = embed_layer.to(device=device, dtype=torch_dtype)

        # 8. 设置为 eval 模式
        model.eval()
        embed_layer.eval()

        # 9. 设置实例属性
        self._model = model
        self._embed_layer = embed_layer
        self._device = device
        self._dtype = dtype
        self._is_loaded = True

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def generate(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs: Any,
    ) -> Any:
        """自回归生成音频码 token。

        Returns
        -------
        torch.Tensor
            生成的音频码 token ids，shape ``[num_vq, generated_len]``。
        """
        if not self._is_loaded or self._model is None or self._embed_layer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_weights() before generate()."
            )

        import torch

        # transformers v5 cache 兼容性回退需捕获的异常类型。显式纳入 CUDA OOM
        # （torch.cuda.OutOfMemoryError 本身是 RuntimeError 的子类，此处列出
        # 以表明意图并兼容未来可能的解耦，见 D1-2）。
        _cache_retry_exc: tuple = (RuntimeError,)
        _oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if isinstance(_oom_type, type) and issubclass(_oom_type, BaseException):
            _cache_retry_exc = (RuntimeError, _oom_type)

        repetition_penalty: float = kwargs.get("repetition_penalty", 1.3)
        spk_emb_pos: int | None = kwargs.get("spk_emb_pos", None)
        eos_token_id: int | None = kwargs.get("eos_token_id", None)

        device = self._device
        num_vq = self._num_vq

        # 通过 DualEmbedding 将 token_ids 转为嵌入
        input_ids = token_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        inputs_embeds = self._embed_layer(input_ids)

        # 说话人嵌入条件
        if speaker_embedding is not None:
            inputs_embeds = self._apply_speaker_embedding(
                inputs_embeds, speaker_embedding, spk_emb_pos
            )

        # 自回归循环
        generated_tokens: list[torch.Tensor] = []
        past_key_values: Any = None
        cur_embeds = inputs_embeds
        finished = False

        for step in range(max_new_tokens):
            if finished:
                break

            with torch.no_grad():
                try:
                    outputs = self._model(
                        inputs_embeds=cur_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                except _cache_retry_exc:
                    # transformers v5 cache 兼容性回退：禁用 KV cache 重试
                    past_key_values = None
                    outputs = self._model(
                        inputs_embeds=cur_embeds,
                        use_cache=False,
                    )
            logits = outputs.logits
            past_key_values = getattr(outputs, "past_key_values", None)

            next_logits = logits[:, -1, :]

            # 对每个 VQ 组分别采样
            new_tokens = []
            for vq_idx in range(num_vq):
                vq_logits = next_logits.clone()
                audio_start = self._num_text_tokens + vq_idx * self._num_audio_tokens
                audio_end = audio_start + self._num_audio_tokens
                mask = torch.full_like(vq_logits, float("-inf"))
                mask[:, audio_start:audio_end] = 0.0
                vq_logits = vq_logits + mask

                if temperature != 1.0:
                    vq_logits = vq_logits / temperature

                vq_logits = self._top_k_filtering(vq_logits, top_k)
                vq_logits = self._top_p_filtering(vq_logits, top_p)

                if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                    generated_ids = torch.cat(generated_tokens, dim=-1)
                    vq_logits = self._apply_repetition_penalty(
                        vq_logits, generated_ids, repetition_penalty
                    )

                probs = torch.softmax(vq_logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                new_tokens.append(sampled)

            new_token = torch.cat(new_tokens, dim=-1)
            generated_tokens.append(new_token)

            if eos_token_id is not None:
                if (new_token == eos_token_id).any():
                    finished = True

            new_token_embed_input = new_token.unsqueeze(1)
            new_embeds = self._embed_layer(new_token_embed_input)
            cur_embeds = new_embeds

        if not generated_tokens:
            return torch.empty((num_vq, 0), dtype=torch.long, device=device)

        # generated_tokens 中每个元素 shape 为 [batch, num_vq]（batch 通常为 1），
        # 沿 dim=0 拼接得到 [generated_len, num_vq]，再转置为 [num_vq, generated_len]
        # 以匹配 docstring 与声码器（DVAE）期望的 [num_vq, frames] 形状。
        # 此前误用了两次 .T（互相抵消），导致输出为 [generated_len, num_vq]，
        # 见 E4-5。
        result = torch.cat(generated_tokens, dim=0).T
        return result

    def generate_stream(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        stream_batch: int = 24,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """流式生成音频码 token。"""
        if not self._is_loaded or self._model is None or self._embed_layer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_weights() before generate_stream()."
            )

        import torch

        # transformers v5 cache 兼容性回退需捕获的异常类型；显式纳入 CUDA OOM
        # （torch.cuda.OutOfMemoryError 是 RuntimeError 的子类，见 D1-2）。
        _cache_retry_exc: tuple = (RuntimeError,)
        _oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if isinstance(_oom_type, type) and issubclass(_oom_type, BaseException):
            _cache_retry_exc = (RuntimeError, _oom_type)

        max_new_tokens: int = kwargs.get("max_new_tokens", 1024)
        temperature: float = kwargs.get("temperature", 1.0)
        top_p: float = kwargs.get("top_p", 0.9)
        top_k: int = kwargs.get("top_k", 50)
        repetition_penalty: float = kwargs.get("repetition_penalty", 1.3)
        spk_emb_pos: int | None = kwargs.get("spk_emb_pos", None)
        eos_token_id: int | None = kwargs.get("eos_token_id", None)

        device = self._device
        num_vq = self._num_vq

        input_ids = token_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        inputs_embeds = self._embed_layer(input_ids)

        if speaker_embedding is not None:
            inputs_embeds = self._apply_speaker_embedding(
                inputs_embeds, speaker_embedding, spk_emb_pos
            )

        buffer: list[torch.Tensor] = []
        past_key_values: Any = None
        cur_embeds = inputs_embeds
        finished = False

        for step in range(max_new_tokens):
            if finished:
                break

            with torch.no_grad():
                try:
                    outputs = self._model(
                        inputs_embeds=cur_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                except _cache_retry_exc:
                    # transformers v5 cache 兼容性回退：禁用 KV cache 重试
                    past_key_values = None
                    outputs = self._model(
                        inputs_embeds=cur_embeds,
                        use_cache=False,
                    )
            logits = outputs.logits
            past_key_values = getattr(outputs, "past_key_values", None)

            next_logits = logits[:, -1, :]

            new_tokens = []
            for vq_idx in range(num_vq):
                vq_logits = next_logits.clone()
                audio_start = self._num_text_tokens + vq_idx * self._num_audio_tokens
                audio_end = audio_start + self._num_audio_tokens
                mask = torch.full_like(vq_logits, float("-inf"))
                mask[:, audio_start:audio_end] = 0.0
                vq_logits = vq_logits + mask

                if temperature != 1.0:
                    vq_logits = vq_logits / temperature

                vq_logits = self._top_k_filtering(vq_logits, top_k)
                vq_logits = self._top_p_filtering(vq_logits, top_p)

                if repetition_penalty != 1.0 and len(buffer) > 0:
                    generated_ids = torch.cat(buffer, dim=-1)
                    vq_logits = self._apply_repetition_penalty(
                        vq_logits, generated_ids, repetition_penalty
                    )

                probs = torch.softmax(vq_logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                new_tokens.append(sampled)

            new_token = torch.cat(new_tokens, dim=-1)
            buffer.append(new_token)

            if eos_token_id is not None:
                if (new_token == eos_token_id).any():
                    finished = True

            new_token_embed_input = new_token.unsqueeze(1)
            new_embeds = self._embed_layer(new_token_embed_input)
            cur_embeds = new_embeds

            if len(buffer) >= stream_batch:
                # 拼接为 [chunk_len, num_vq] 后转置为 [num_vq, chunk_len]，
                # 与 generate() 一致地匹配声码器期望的 [num_vq, frames] 形状
                # （见 E4-5：此前误用两次 .T 互相抵消）。
                chunk = torch.cat(buffer, dim=0).T
                yield chunk
                buffer = []

        if buffer:
            chunk = torch.cat(buffer, dim=0).T
            yield chunk

    # ------------------------------------------------------------------
    # ChatTTS 特有方法
    # ------------------------------------------------------------------
    def _apply_speaker_embedding(
        self,
        emb: Any,
        speaker_embedding: Any,
        spk_emb_pos: int | None,
    ) -> Any:
        """将说话人嵌入应用到嵌入序列的指定位置。"""
        import torch

        spk = speaker_embedding.to(emb.device, emb.dtype)
        spk = spk / (spk.norm(dim=-1, keepdim=True) + 1e-9)

        # 说话人嵌入维度（如 ChatTTS 的 256）可能与 hidden_size（如 768）
        # 不一致，torch.where 要求最后一维相同。通过补零或截断对齐。
        hidden_dim = emb.shape[-1]
        if spk.shape[-1] != hidden_dim:
            if spk.shape[-1] < hidden_dim:
                pad_size = hidden_dim - spk.shape[-1]
                spk = torch.nn.functional.pad(spk, (0, pad_size))
            else:
                spk = spk[..., :hidden_dim]

        pos = 0 if spk_emb_pos is None else spk_emb_pos

        batch, seq_len = emb.shape[0], emb.shape[1]
        pos_ids = torch.arange(seq_len, device=emb.device).unsqueeze(0)
        pos_mask = (pos_ids == pos)

        if spk.dim() == 1:
            spk = spk.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        elif spk.dim() == 2:
            spk = spk.unsqueeze(1).expand(-1, seq_len, -1)

        mask = pos_mask.unsqueeze(-1)
        emb = torch.where(mask, spk, emb)
        return emb
