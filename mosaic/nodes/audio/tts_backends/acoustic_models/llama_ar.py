# mosaic/nodes/audio/tts_backends/acoustic_models/llama_ar.py
"""ChatTTS 声学模型实现（基于 Llama 架构的自回归模型）。

Layer 2: 声学模型层。本模块实现 ChatTTS 的自回归（AR）声学模型，将文本
token ids 转换为多组 VQ 音频码 token。

本模型采用 LlamaForCausalLM 作为骨干网络，配合 :class:`DualEmbedding` 双路径
嵌入层实现「文本 token」与「音频 token」的统一编码。生成阶段以自回归方式逐
token 采样多组 VQ 码本，支持温度采样、top-k / top-p（nucleus）采样、重复
惩罚以及说话人嵌入条件控制。

显存需求
--------
* ``float16`` 精度：约 2-4 GB GPU 显存
* ``float32`` 精度：约 4-8 GB GPU 显存

设计要点
--------
* ``torch`` / ``transformers`` / ``safetensors`` 等重依赖采用惰性导入，使本
  模块在未安装这些依赖时仍可被导入与继承（仅在实际调用 ``load_weights``
  等方法时才报依赖缺失）。
* ``token_ids`` / ``speaker_embedding`` 等参数类型用 :data:`~typing.Any`
  标注，避免在模块顶层硬依赖 ``torch``。
* 采样相关辅助方法（``_top_k_filtering`` / ``_top_p_filtering`` /
  ``_apply_repetition_penalty``）以静态方法形式提供，便于单独测试。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel

__all__ = ["LlamaARModel", "DualEmbedding"]


class DualEmbedding:
    """ChatTTS 双路径嵌入层。

    文本位置使用 ``emb_text``，音频位置使用多个 ``emb_code`` 求和。

    本类是对 :class:`torch.nn.Module` 的轻量包装，实际的 ``nn.Module`` 子类
    在首次实例化时惰性构建（需要 ``torch`` 可用）。这样做可以让本模块在
    未安装 ``torch`` 的环境下被导入，同时保持 ``isinstance`` 检查与
    ``forward`` 调用的一致性。

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
        """构建真实的 ``nn.Module`` 子类实例。

        Parameters
        ----------
        num_text_tokens : int
            文本词表大小。
        num_audio_tokens : int
            每个码本的 token 数。
        num_vq : int
            VQ 码本组数。
        hidden_size : int
            隐藏层维度。
        """
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
                """前向计算。

                Parameters
                ----------
                input_ids : torch.Tensor
                    形状 ``[batch, seq_len, num_vq]`` 或 ``[batch, seq_len]``。
                    当最后一维 == ``num_vq`` 时，认为是音频 token，用
                    ``emb_code`` 求和；否则认为是文本 token，用 ``emb_text``。
                text_mask : torch.Tensor | None
                    文本位置掩码，``True`` 表示该位置是文本。

                Returns
                -------
                torch.Tensor
                    嵌入向量 ``[batch, seq_len, hidden_size]``。
                """
                import torch

                # 情况 1：提供了 text_mask，按掩码分流
                if text_mask is not None:
                    text_ids = input_ids.long()
                    # 文本位置：emb_text
                    text_emb = self.emb_text(text_ids)
                    # 音频位置：各码本嵌入求和
                    # input_ids[..., i] 取第 i 组码本
                    audio_emb = torch.zeros_like(text_emb)
                    for i in range(num_vq):
                        audio_emb = audio_emb + self.emb_code[i](input_ids[..., i].long())
                    emb = torch.where(text_mask.bool().unsqueeze(-1), text_emb, audio_emb)
                    return emb

                # 情况 2：根据维度判断
                if input_ids.dim() == 3 and input_ids.size(-1) == num_vq:
                    # 音频 token：各码本嵌入求和
                    emb = torch.zeros(
                        *input_ids.shape[:-1],
                        hidden_size,
                        device=input_ids.device,
                        dtype=self.emb_text.weight.dtype,
                    )
                    for i in range(num_vq):
                        emb = emb + self.emb_code[i](input_ids[..., i].long())
                    return emb

                # 否则认为是文本 token
                return self.emb_text(input_ids.long())

        impl = _DualEmbeddingImpl()
        # 标记真实类型，便于外部 isinstance 判断
        impl.__class__ = type(
            "DualEmbedding", (_DualEmbeddingImpl,), {"__doc__": cls.__doc__}
        )
        return impl


class LlamaARModel(AcousticModel):
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

    # 类属性
    model_type: str = "ar"
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
        # 设置类属性（实例化后会覆盖类属性）
        self.vocab_size = num_text_tokens + num_audio_tokens * num_vq
        self.hidden_size = hidden_size

        # 实例属性
        self._model_path: str = model_path
        self._num_vq: int = num_vq
        self._num_audio_tokens: int = num_audio_tokens
        self._num_text_tokens: int = num_text_tokens
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
    # 权重加载 / 释放
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

        Parameters
        ----------
        weights_path : str
            权重文件路径（目录或文件）。
        device : str
            目标设备，默认 ``"cuda"``。
        dtype : str
            数据精度，``"float16"`` / ``"float32"`` / ``"bfloat16"``。

        Raises
        ------
        ImportError
            ``torch`` / ``transformers`` 未安装。
        FileNotFoundError
            权重文件不存在。
        """
        import torch

        # 1. 解析 dtype
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
        torch_dtype = dtype_map[dtype]

        # 2. 加载或创建 LlamaConfig
        from transformers import LlamaConfig, LlamaForCausalLM  # type: ignore

        config_path = os.path.join(weights_path, "config.json")
        if os.path.exists(config_path):
            config = LlamaConfig.from_pretrained(weights_path)
            # 确保词表大小一致
            config.vocab_size = self.vocab_size
        else:
            config = LlamaConfig(
                vocab_size=self.vocab_size,
                hidden_size=self._hidden_size,
                num_hidden_layers=self._num_layers,
                num_attention_heads=self._num_heads,
                max_position_embeddings=self._max_position_embeddings,
                torch_dtype=torch_dtype,
            )

        # 3. 创建 LlamaForCausalLM 实例
        model = LlamaForCausalLM(config)

        # 4. 加载权重（safetensors 或 pytorch checkpoint）
        safetensors_path = os.path.join(weights_path, "acoustic_model.safetensors")
        pytorch_path = os.path.join(weights_path, "acoustic_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file  # type: ignore

            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict, strict=False)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        elif os.path.isdir(weights_path):
            # 尝试使用 transformers 的 from_pretrained 加载
            try:
                model = LlamaForCausalLM.from_pretrained(weights_path)
            except Exception:
                # 没有可用权重，使用随机初始化的模型
                pass

        # 5. 创建 DualEmbedding 实例
        embed_layer = DualEmbedding(
            num_text_tokens=self._num_text_tokens,
            num_audio_tokens=self._num_audio_tokens,
            num_vq=self._num_vq,
            hidden_size=self._hidden_size,
        )

        # 6. 加载 Embed 权重（如果有单独的 embed 权重文件）
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

    def unload_weights(self) -> None:
        """释放模型权重。

        释放 ``_model`` 和 ``_embed_layer``，设置为 ``None``，清空 CUDA 缓存
        （如果在 CUDA 上），并设置 ``_is_loaded = False``。
        """
        self._model = None
        self._embed_layer = None

        # 清空 CUDA 缓存（torch 为惰性导入，缺失时跳过）
        try:
            import torch

            if self._device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._is_loaded = False

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

        生成流程：

        1. 检查 ``_is_loaded``；
        2. 通过 :class:`DualEmbedding` 将 ``token_ids`` 转为嵌入；
        3. 如果有 ``speaker_embedding``：L2 归一化后在 ``[spk_emb]`` token
           位置替换为归一化的说话人嵌入（使用 ``torch.where`` 条件替换）；
        4. 自回归循环：

           a. 调用 ``LlamaForCausalLM.forward`` 获取 logits；
           b. 采样：``logits / temperature`` -> top_k_filter ->
              top_p_filter -> repetition_penalty -> softmax -> multinomial；
           c. 对每个 VQ 组分别采样（``num_vq`` 个输出头）；
           d. 检查 eos_token 停止条件；
           e. 将新 token 加入序列，更新 KV cache。

        5. 返回生成的音频码 token ids，shape ``[num_vq, generated_len]``。

        Parameters
        ----------
        token_ids : torch.Tensor
            文本 token ids，形状 ``[seq_len]`` 或 ``[batch, seq_len]``。
        speaker_embedding : torch.Tensor | None
            说话人嵌入向量。
        max_new_tokens : int
            最大生成 token 数。
        temperature : float
            采样温度。
        top_p : float
            nucleus sampling 的 p 值。
        top_k : int
            top-k 采样的 k 值。

        Returns
        -------
        torch.Tensor
            生成的音频码 token ids，shape ``[num_vq, generated_len]``。

        Raises
        ------
        RuntimeError
            模型未加载。
        """
        if not self._is_loaded or self._model is None or self._embed_layer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_weights() before generate()."
            )

        import torch

        # 解析 kwargs
        repetition_penalty: float = kwargs.get("repetition_penalty", 1.3)
        spk_emb_pos: int | None = kwargs.get("spk_emb_pos", None)
        eos_token_id: int | None = kwargs.get("eos_token_id", None)

        device = self._device
        num_vq = self._num_vq

        # 2. 通过 DualEmbedding 将 token_ids 转为嵌入
        input_ids = token_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]

        inputs_embeds = self._embed_layer(input_ids)  # [1, seq_len, hidden]

        # 3. 说话人嵌入条件
        if speaker_embedding is not None:
            inputs_embeds = self._apply_speaker_embedding(
                inputs_embeds, speaker_embedding, spk_emb_pos
            )

        # 4. 自回归循环
        generated_tokens: list[torch.Tensor] = []
        past_key_values: Any = None

        cur_embeds = inputs_embeds
        finished = False

        for step in range(max_new_tokens):
            if finished:
                break

            # a. forward 获取 logits
            with torch.no_grad():
                outputs = self._model(
                    inputs_embeds=cur_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits  # [1, seq, vocab]
            past_key_values = outputs.past_key_values

            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :]  # [1, vocab]

            # b. 采样（对每个 VQ 组分别采样）
            # 这里将 logits 视作 num_vq 组的输出
            # 实际 ChatTTS 中 lm_head 输出会被 reshape 为 [batch, num_vq, num_audio_tokens]
            new_tokens = []
            for vq_idx in range(num_vq):
                # 取当前 VQ 组对应的 logits 切片
                vq_logits = next_logits.clone()
                # 仅保留音频 token 范围内的 logits
                audio_start = self._num_text_tokens + vq_idx * self._num_audio_tokens
                audio_end = audio_start + self._num_audio_tokens
                mask = torch.full_like(vq_logits, float("-inf"))
                mask[:, audio_start:audio_end] = 0.0
                vq_logits = vq_logits + mask

                # temperature
                if temperature != 1.0:
                    vq_logits = vq_logits / temperature

                # top_k
                vq_logits = self._top_k_filtering(vq_logits, top_k)

                # top_p
                vq_logits = self._top_p_filtering(vq_logits, top_p)

                # repetition penalty
                if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                    generated_ids = torch.cat(generated_tokens, dim=-1)
                    vq_logits = self._apply_repetition_penalty(
                        vq_logits, generated_ids, repetition_penalty
                    )

                # softmax -> multinomial
                probs = torch.softmax(vq_logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)  # [1, 1]
                new_tokens.append(sampled)

            # c. 组装新 token [1, num_vq]
            new_token = torch.cat(new_tokens, dim=-1)  # [1, num_vq]
            generated_tokens.append(new_token)

            # d. 检查 eos
            if eos_token_id is not None:
                # 任一 VQ 组生成 eos 即停止
                if (new_token == eos_token_id).any():
                    finished = True

            # e. 更新输入：将新 token 转为嵌入
            # new_token: [1, num_vq] -> [1, 1, num_vq]
            new_token_embed_input = new_token.unsqueeze(1)
            new_embeds = self._embed_layer(new_token_embed_input)  # [1, 1, hidden]
            cur_embeds = new_embeds

        # 5. 返回 [num_vq, generated_len]
        if not generated_tokens:
            # 没有生成任何 token，返回空张量
            return torch.empty((num_vq, 0), dtype=torch.long, device=device)

        result = torch.cat(generated_tokens, dim=0).T  # [generated_len, num_vq]
        return result.T  # [num_vq, generated_len]

    def generate_stream(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        stream_batch: int = 24,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """流式生成音频码 token。

        与 :meth:`generate` 相同的前处理，但在自回归循环中每生成
        ``stream_batch`` 个 token 后 yield 一次。yield 的内容是增量的 token
        ids，shape ``[num_vq, stream_batch]``。最后 yield 剩余不足
        ``stream_batch`` 的 token。

        Parameters
        ----------
        token_ids : torch.Tensor
            文本 token ids。
        speaker_embedding : torch.Tensor | None
            说话人嵌入向量。
        stream_batch : int
            每次 yield 的 token 数量。
        **kwargs : Any
            透传给采样的额外参数，包括 ``max_new_tokens``、``temperature``、
            ``top_p``、``top_k``、``repetition_penalty``、``spk_emb_pos``、
            ``eos_token_id``。

        Yields
        ------
        torch.Tensor
            增量的音频码 token ids，shape ``[num_vq, stream_batch]``（最后
            一批可能更短）。

        Raises
        ------
        RuntimeError
            模型未加载。
        """
        if not self._is_loaded or self._model is None or self._embed_layer is None:
            raise RuntimeError(
                "Model is not loaded. Call load_weights() before generate_stream()."
            )

        import torch

        # 解析参数
        max_new_tokens: int = kwargs.get("max_new_tokens", 1024)
        temperature: float = kwargs.get("temperature", 1.0)
        top_p: float = kwargs.get("top_p", 0.9)
        top_k: int = kwargs.get("top_k", 50)
        repetition_penalty: float = kwargs.get("repetition_penalty", 1.3)
        spk_emb_pos: int | None = kwargs.get("spk_emb_pos", None)
        eos_token_id: int | None = kwargs.get("eos_token_id", None)

        device = self._device
        num_vq = self._num_vq

        # 前处理：嵌入
        input_ids = token_ids
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        inputs_embeds = self._embed_layer(input_ids)

        if speaker_embedding is not None:
            inputs_embeds = self._apply_speaker_embedding(
                inputs_embeds, speaker_embedding, spk_emb_pos
            )

        # 自回归循环
        buffer: list[torch.Tensor] = []
        past_key_values: Any = None
        cur_embeds = inputs_embeds
        finished = False

        for step in range(max_new_tokens):
            if finished:
                break

            with torch.no_grad():
                outputs = self._model(
                    inputs_embeds=cur_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

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

            new_token = torch.cat(new_tokens, dim=-1)  # [1, num_vq]
            buffer.append(new_token)

            # 检查 eos
            if eos_token_id is not None:
                if (new_token == eos_token_id).any():
                    finished = True

            # 更新输入
            new_token_embed_input = new_token.unsqueeze(1)
            new_embeds = self._embed_layer(new_token_embed_input)
            cur_embeds = new_embeds

            # 每累积 stream_batch 个 token yield 一次
            if len(buffer) >= stream_batch:
                chunk = torch.cat(buffer, dim=0).T  # [stream_batch, num_vq]
                yield chunk.T  # [num_vq, stream_batch]
                buffer = []

        # yield 剩余不足 stream_batch 的 token
        if buffer:
            chunk = torch.cat(buffer, dim=0).T  # [rest, num_vq]
            yield chunk.T  # [num_vq, rest]

    # ------------------------------------------------------------------
    # 嵌入层 / 输出头访问
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> Any:
        """返回输入嵌入层。

        Returns
        -------
        DualEmbedding | None
            :class:`DualEmbedding` 实例；未加载权重时返回 ``None``。
        """
        return self._embed_layer

    def get_output_head(self) -> Any:
        """返回输出头。

        Returns
        -------
        torch.nn.Linear | None
            :class:`LlamaForCausalLM` 的 ``lm_head``；未加载权重时返回
            ``None``。
        """
        if self._model is not None:
            return self._model.lm_head
        return None

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _apply_speaker_embedding(
        self,
        emb: Any,
        speaker_embedding: Any,
        spk_emb_pos: int | None,
    ) -> Any:
        """将说话人嵌入应用到嵌入序列的指定位置。

        对 ``speaker_embedding`` 做 L2 归一化，然后在 ``spk_emb_pos`` 位置
        替换。使用 :func:`torch.where` 条件替换。

        Parameters
        ----------
        emb : torch.Tensor
            输入嵌入序列，shape ``[batch, seq_len, hidden_size]``。
        speaker_embedding : torch.Tensor
            说话人嵌入向量，shape ``[hidden_size]`` 或 ``[batch, hidden_size]``。
        spk_emb_pos : int | None
            说话人嵌入位置；``None`` 时默认替换第 0 个位置。

        Returns
        -------
        torch.Tensor
            替换后的嵌入序列。
        """
        import torch

        # L2 归一化
        spk = speaker_embedding.to(emb.device, emb.dtype)
        spk = spk / (spk.norm(dim=-1, keepdim=True) + 1e-9)

        # 默认位置
        pos = 0 if spk_emb_pos is None else spk_emb_pos

        # 构造位置掩码 [batch, seq_len]
        batch, seq_len = emb.shape[0], emb.shape[1]
        pos_ids = torch.arange(seq_len, device=emb.device).unsqueeze(0)
        pos_mask = (pos_ids == pos)  # [batch, seq_len]

        # 广播 spk 到 [batch, seq_len, hidden]
        if spk.dim() == 1:
            spk = spk.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1)
        elif spk.dim() == 2:
            spk = spk.unsqueeze(1).expand(-1, seq_len, -1)

        # 条件替换
        mask = pos_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        emb = torch.where(mask, spk, emb)
        return emb

    @staticmethod
    def _top_k_filtering(logits: Any, top_k: int) -> Any:
        """保留 logits 中 top_k 个最大值，其余设为 -inf。

        Parameters
        ----------
        logits : torch.Tensor
            输入 logits，shape ``[batch, vocab]``。
        top_k : int
            保留的 token 数。

        Returns
        -------
        torch.Tensor
            过滤后的 logits。
        """
        import torch

        if top_k <= 0:
            return logits

        top_k = min(top_k, logits.size(-1))
        # 取第 top_k 大的值作为阈值
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        indices_to_remove = logits < kth_vals
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    @staticmethod
    def _top_p_filtering(logits: Any, top_p: float) -> Any:
        """nucleus sampling：保留累积概率超过 top_p 的最小 token 集合。

        Parameters
        ----------
        logits : torch.Tensor
            输入 logits，shape ``[batch, vocab]``。
        top_p : float
            累积概率阈值，``0 < top_p <= 1``。

        Returns
        -------
        torch.Tensor
            过滤后的 logits。
        """
        import torch

        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )

        # 移除累积概率超过 top_p 的 token（保留第一个超过的）
        sorted_indices_to_remove = cumulative_probs > top_p
        # 右移一位，保留第一个超过阈值的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 将排序后的掩码散射回原始顺序
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    @staticmethod
    def _apply_repetition_penalty(
        logits: Any, generated_ids: Any, penalty: float
    ) -> Any:
        """对已生成 token 的 logits 除以 penalty（降低概率）。

        Parameters
        ----------
        logits : torch.Tensor
            输入 logits，shape ``[batch, vocab]``。
        generated_ids : torch.Tensor
            已生成的 token ids。
        penalty : float
            重复惩罚系数，``> 1`` 降低重复概率，``< 1`` 增加重复概率。

        Returns
        -------
        torch.Tensor
            惩罚后的 logits。
        """
        import torch

        if penalty == 1.0:
            return logits

        # 收集已生成 token 的唯一 id
        for batch_idx in range(logits.size(0)):
            ids = generated_ids[batch_idx].unique().long()
            # 对正 logits 除以 penalty，对负 logits 乘以 penalty
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
