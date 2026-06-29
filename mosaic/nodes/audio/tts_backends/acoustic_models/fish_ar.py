# mosaic/nodes/audio/tts_backends/acoustic_models/fish_ar.py
"""Fish Speech 基于 Llama 的自回归声学模型。

文件路径: mosaic/nodes/audio/tts_backends/acoustic_models/fish_ar.py

Layer 2: 声学模型层。将文本 token ids 转换为离散音频 codec token。

与 ChatTTS 的 :class:`LlamaARModel` 相比，Fish Speech 的差异在于：

* **统一 Embed**：文本和音频共用一个 ``Embedding(total_vocab, hidden)``
  （ChatTTS 使用双路径 ``emb_text`` + ``emb_code`` 求和）。
* **语音克隆**：通过参考音频的 codec token ids 拼接到输入序列前部实现，
  而非 ChatTTS 的 ``spk_emb`` 向量替换。
* **单路输出**：LLaMA 输出的 logits 直接对应统一词表，只需取音频范围的
  logits 采样（ChatTTS 需要对每个 VQ 组分别采样）。

显存需求
--------
* ``float16`` 精度：约 2-4 GB GPU 显存
* ``float32`` 精度：约 4-8 GB GPU 显存
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from mosaic.nodes.audio.tts_backends.acoustic_models.llama_ar import (
    LlamaARModelBase,
)

__all__ = ["FishLlamaARModel", "UnifiedEmbedding"]


class UnifiedEmbedding:
    """Fish Speech 统一嵌入层。

    文本 token 和音频 codec token 共用同一个 ``Embedding`` 查找表。
    文本 token id 范围 ``[0, text_vocab_size)``，
    音频 token id 范围 ``[text_vocab_size, total_vocab_size)``。

    本类是对 :class:`torch.nn.Module` 的轻量包装，实际的 ``nn.Module`` 子类
    在首次实例化时惰性构建（需要 ``torch`` 可用）。
    """

    def __new__(
        cls,
        total_vocab_size: int,
        hidden_size: int,
    ) -> "UnifiedEmbedding":
        """构建真实的 ``nn.Module`` 子类实例。"""
        import torch.nn as nn

        class _UnifiedEmbeddingImpl(nn.Module):
            """Fish Speech 统一嵌入层的真实实现。"""

            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(total_vocab_size, hidden_size)

            def forward(self, input_ids: Any) -> Any:
                """前向计算。

                Parameters
                ----------
                input_ids : torch.Tensor
                    token ids，shape ``[batch, seq_len]``。

                Returns
                -------
                torch.Tensor
                    嵌入向量，shape ``[batch, seq_len, hidden_size]``。
                """
                # clamp 防止字符级回退产生的越界 id 导致 Embedding 查表报错
                safe_ids = input_ids.long().clamp(min=0, max=total_vocab_size - 1)
                return self.emb(safe_ids)

        impl = _UnifiedEmbeddingImpl()
        impl.__class__ = type(
            "UnifiedEmbedding", (_UnifiedEmbeddingImpl,), {"__doc__": cls.__doc__}
        )
        return impl


class FishLlamaARModel(LlamaARModelBase):
    """Fish Speech 基于 Llama 的自回归声学模型。

    使用 :class:`transformers.LlamaForCausalLM` 作为骨干网络，配合
    :class:`UnifiedEmbedding` 统一嵌入层，将文本 token ids 自回归地转换为
    音频 codec token。

    语音克隆通过将参考音频的 codec token ids 拼接到输入序列前部实现，
    模型以参考音频为条件生成目标音频。

    Attributes
    ----------
    model_type : str
        模型类型，固定为 ``"ar"``。
    vocab_size : int
        统一词表大小，等于 ``text_vocab_size + audio_vocab_size``。
    hidden_size : int
        隐藏层维度。
    """

    vocab_size: int = 0
    hidden_size: int = 0

    def __init__(
        self,
        model_path: str,
        text_vocab_size: int = 0,
        audio_vocab_size: int = 0,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        max_position_embeddings: int = 2048,
        use_flash_attention: bool = True,
        codec_type: str = "dac",
    ) -> None:
        """初始化 FishLlamaARModel。

        Parameters
        ----------
        model_path : str
            Fish Speech 模型路径（包含 ``config.json`` 与权重文件）。
        text_vocab_size : int
            文本词表大小。
        audio_vocab_size : int
            音频 codec 词表大小。
        hidden_size : int
            隐藏层维度，默认 ``1024``。
        num_heads : int
            注意力头数，默认 ``16``。
        num_layers : int
            Transformer 层数，默认 ``24``。
        max_position_embeddings : int
            最大位置编码长度，默认 ``2048``。
        use_flash_attention : bool
            是否使用 Flash Attention，默认 ``True``。
        codec_type : str
            音频编码器类型，``"dac"`` / ``"encodec"`` / ``"snac"``，默认 ``"dac"``。
        """
        super().__init__(
            model_path=model_path,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_position_embeddings=max_position_embeddings,
            use_flash_attention=use_flash_attention,
        )

        # Fish Speech 特有属性
        self.vocab_size = text_vocab_size + audio_vocab_size
        self.hidden_size = hidden_size
        self._text_vocab_size: int = text_vocab_size
        self._audio_vocab_size: int = audio_vocab_size
        self._codec_type: str = codec_type
        self._audio_encoder: Any = None

    # ------------------------------------------------------------------
    # 权重加载
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载模型权重。

        加载步骤：

        1. 解析 dtype 字符串为 torch dtype；
        2. 创建 :class:`LlamaConfig` 并实例化 :class:`LlamaForCausalLM`；
        3. 加载 Transformer 层权重；
        4. 创建 :class:`UnifiedEmbedding` 并加载权重；
        5. 移动到 device/dtype；
        6. 设置为 eval 模式。
        """
        import torch

        # 1. 解析 dtype
        torch_dtype = self._parse_dtype(dtype)

        # 2-3. 创建 LlamaConfig 和 LlamaForCausalLM
        from transformers import LlamaForCausalLM  # type: ignore

        config = self._create_llama_config(
            weights_path, torch_dtype, self.vocab_size
        )
        model = LlamaForCausalLM(config)

        # 4. 加载 Transformer 权重
        self._load_llama_weights(model, weights_path)

        # 5. 创建 UnifiedEmbedding 实例
        embed_layer = UnifiedEmbedding(
            total_vocab_size=self.vocab_size,
            hidden_size=self._hidden_size,
        )

        # 6. 加载 Embed 权重
        embed_path = os.path.join(weights_path, "embed.safetensors")
        if not os.path.exists(embed_path):
            embed_path = os.path.join(weights_path, "text_frontend.safetensors")
        if os.path.exists(embed_path):
            from safetensors.torch import load_file  # type: ignore

            embed_state = load_file(embed_path)
            # 尝试匹配统一嵌入权重
            mapped_state: dict[str, Any] = {}
            for key, val in embed_state.items():
                if "embed_tokens" in key or "emb" in key:
                    mapped_state["emb.weight"] = val
            if mapped_state:
                embed_layer.load_state_dict(mapped_state, strict=False)

        # 7. 移动到 device 和 dtype
        model = model.to(device=device, dtype=torch_dtype)
        embed_layer = embed_layer.to(device=device, dtype=torch_dtype)

        # 8. eval 模式
        model.eval()
        embed_layer.eval()

        # 9. 设置属性
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
        """自回归生成音频 codec token。

        Parameters
        ----------
        token_ids : torch.Tensor
            输入 token ids，shape ``[1, seq_len]`` 或 ``[seq_len]``。
        speaker_embedding : torch.Tensor | None
            Fish Speech 中是参考音频的 codec token ids（``[1, ref_len]``）。
            为 ``None`` 时不使用语音克隆。
        max_new_tokens : int
            最大生成 token 数。
        temperature : float
            采样温度。
        top_p : float
            nucleus sampling 参数。
        top_k : int
            top-k sampling 参数。

        Returns
        -------
        torch.Tensor
            生成的音频 codec token ids，shape ``[1, generated_len]``。
            token id 已减去 ``text_vocab_size`` 偏移，范围 ``[0, audio_vocab_size)``。
        """
        if not self._is_loaded or self._model is None or self._embed_layer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_weights() before generate()."
            )

        import torch

        repetition_penalty: float = kwargs.get("repetition_penalty", 1.3)
        eos_token_id: int | None = kwargs.get("eos_token_id", None)
        text_stop_threshold: int = kwargs.get("text_stop_threshold", 5)

        device = self._device

        input_ids = token_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # 语音克隆：将参考音频的 codec token ids 拼接到输入前
        if speaker_embedding is not None:
            ref_ids = speaker_embedding
            if ref_ids.dim() == 1:
                ref_ids = ref_ids.unsqueeze(0)
            # 参考音频的 token ids 已经是音频范围（带偏移）
            input_ids = torch.cat([ref_ids, input_ids], dim=-1)

        # 通过统一 Embedding 转为嵌入
        inputs_embeds = self._embed_layer(input_ids)

        # 自回归循环
        generated_tokens: list[torch.Tensor] = []
        past_key_values: Any = None
        cur_embeds = inputs_embeds
        finished = False
        consecutive_text_count = 0

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
                except RuntimeError:
                    # transformers v5 cache 兼容性回退：禁用 KV cache 重试
                    past_key_values = None
                    outputs = self._model(
                        inputs_embeds=cur_embeds,
                        use_cache=False,
                    )
            logits = outputs.logits
            past_key_values = getattr(outputs, "past_key_values", None)

            next_logits = logits[:, -1, :]

            # 只取音频范围的 logits
            audio_start = self._text_vocab_size
            audio_logits = next_logits[:, audio_start:]

            if temperature != 1.0:
                audio_logits = audio_logits / temperature

            audio_logits = self._top_k_filtering(audio_logits, top_k)
            audio_logits = self._top_p_filtering(audio_logits, top_p)

            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                # generated_ids 是统一词表中的绝对 id（含 text_vocab 偏移），
                # 但 audio_logits 已被切片到音频范围，需减去偏移以匹配索引
                generated_ids = torch.cat(generated_tokens, dim=-1) - self._text_vocab_size
                audio_logits = self._apply_repetition_penalty(
                    audio_logits, generated_ids, repetition_penalty
                )

            probs = torch.softmax(audio_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)

            # 采样结果 + 偏移 = 统一词表中的 id
            unified_id = sampled + self._text_vocab_size
            generated_tokens.append(unified_id)

            # 停止条件检查
            if eos_token_id is not None and (unified_id == eos_token_id).any():
                finished = True
                break

            # 连续 N 个 token 都在文本范围（异常情况），停止生成
            if (unified_id < self._text_vocab_size).any():
                consecutive_text_count += 1
                if consecutive_text_count >= text_stop_threshold:
                    finished = True
            else:
                consecutive_text_count = 0

            # 准备下一步输入（unified_id 已为 [batch, 1]，无需额外 unsqueeze）
            next_embed = self._embed_layer(unified_id)
            cur_embeds = next_embed

        if not generated_tokens:
            return torch.empty((1, 0), dtype=torch.long, device=device)

        result = torch.cat(generated_tokens, dim=-1).unsqueeze(0)
        # 减去偏移，得到实际 codec index
        result = result - self._text_vocab_size
        return result

    def generate_stream(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        stream_batch: int = 24,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """流式生成音频 codec token。

        Parameters
        ----------
        token_ids : torch.Tensor
            输入 token ids。
        speaker_embedding : torch.Tensor | None
            参考音频的 codec token ids（语音克隆）。
        stream_batch : int
            每多少个 token yield 一次。

        Yields
        ------
        torch.Tensor
            增量的音频 codec token ids，shape ``[1, chunk_len]``。
        """
        if not self._is_loaded or self._model is None or self._embed_layer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_weights() before generate_stream()."
            )

        import torch

        max_new_tokens: int = kwargs.get("max_new_tokens", 1024)
        temperature: float = kwargs.get("temperature", 1.0)
        top_p: float = kwargs.get("top_p", 0.9)
        top_k: int = kwargs.get("top_k", 50)
        repetition_penalty: float = kwargs.get("repetition_penalty", 1.3)
        eos_token_id: int | None = kwargs.get("eos_token_id", None)
        text_stop_threshold: int = kwargs.get("text_stop_threshold", 5)

        device = self._device

        input_ids = token_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if speaker_embedding is not None:
            ref_ids = speaker_embedding
            if ref_ids.dim() == 1:
                ref_ids = ref_ids.unsqueeze(0)
            input_ids = torch.cat([ref_ids, input_ids], dim=-1)

        inputs_embeds = self._embed_layer(input_ids)

        buffer: list[torch.Tensor] = []
        past_key_values: Any = None
        cur_embeds = inputs_embeds
        finished = False
        consecutive_text_count = 0

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
                except RuntimeError:
                    # transformers v5 cache 兼容性回退：禁用 KV cache 重试
                    past_key_values = None
                    outputs = self._model(
                        inputs_embeds=cur_embeds,
                        use_cache=False,
                    )
            logits = outputs.logits
            past_key_values = getattr(outputs, "past_key_values", None)

            next_logits = logits[:, -1, :]

            audio_start = self._text_vocab_size
            audio_logits = next_logits[:, audio_start:]

            if temperature != 1.0:
                audio_logits = audio_logits / temperature

            audio_logits = self._top_k_filtering(audio_logits, top_k)
            audio_logits = self._top_p_filtering(audio_logits, top_p)

            if repetition_penalty != 1.0 and len(buffer) > 0:
                # buffer 中是统一词表绝对 id，需减去偏移匹配切片后的 audio_logits
                generated_ids = torch.cat(buffer, dim=-1) - self._text_vocab_size
                audio_logits = self._apply_repetition_penalty(
                    audio_logits, generated_ids, repetition_penalty
                )

            probs = torch.softmax(audio_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)

            unified_id = sampled + self._text_vocab_size
            buffer.append(unified_id)

            # 停止条件
            if eos_token_id is not None and (unified_id == eos_token_id).any():
                finished = True

            if (unified_id < self._text_vocab_size).any():
                consecutive_text_count += 1
                if consecutive_text_count >= text_stop_threshold:
                    finished = True
            else:
                consecutive_text_count = 0

            # unified_id 已为 [batch, 1]，无需额外 unsqueeze
            next_embed = self._embed_layer(unified_id)
            cur_embeds = next_embed

            if len(buffer) >= stream_batch:
                chunk = torch.cat(buffer, dim=-1).unsqueeze(0)
                yield chunk - self._text_vocab_size
                buffer = []

        if buffer:
            chunk = torch.cat(buffer, dim=-1).unsqueeze(0)
            yield chunk - self._text_vocab_size

    # ------------------------------------------------------------------
    # Fish Speech 特有方法
    # ------------------------------------------------------------------
    def encode_reference_audio(self, audio: Any) -> Any:
        """将参考音频编码为 codec token ids（用于语音克隆）。

        使用 AudioEncoder（DAC / Encodec / SNAC）将波形编码为离散 token。

        Parameters
        ----------
        audio : AudioData
            参考音频数据。

        Returns
        -------
        torch.Tensor
            codec token ids，已加上 ``text_vocab_size`` 偏移，
            shape ``[1, ref_len]``。

        Raises
        ------
        RuntimeError
            AudioEncoder 未加载或编码失败。
        """
        import torch

        if self._audio_encoder is None:
            raise RuntimeError(
                "AudioEncoder is not loaded. Cannot encode reference audio."
            )

        # 获取波形
        if hasattr(audio, "waveform"):
            waveform = audio.waveform
            sr = getattr(audio, "sample_rate", 24000)
        else:
            waveform = audio
            sr = 24000

        if hasattr(waveform, "cpu"):
            waveform = waveform.cpu()

        # 编码为 codec tokens
        codec_tokens = self._audio_encoder.encode(waveform, sr)

        # 加上偏移，映射到统一词表的音频范围
        codec_tokens = codec_tokens + self._text_vocab_size

        return codec_tokens
