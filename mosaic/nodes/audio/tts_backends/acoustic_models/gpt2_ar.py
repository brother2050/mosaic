# mosaic/nodes/audio/tts_backends/acoustic_models/gpt2_ar.py
"""GPT-SoVITS GPT-2 自回归声学模型。

文件路径: mosaic/nodes/audio/tts_backends/acoustic_models/gpt2_ar.py

Layer 2: 声学模型层。将文本音素 token ids 转换为语义 token ids。

GPT-SoVITS 的 GPT 部分使用 GPT-2 架构自回归生成语义 token。与 ChatTTS /
Fish Speech 的 LLaMA 模型不同，GPT-2 有独立的 transformers API，因此本类
**不继承** :class:`LlamaARModelBase`，直接继承 :class:`AcousticModel`。

双路径 Embedding
----------------
GPT-SoVITS 使用两个独立的 Embedding 层：

* ``text_embedding``：文本音素 → 嵌入向量
* ``semantic_embedding``：语义 token → 嵌入向量（参考音频与生成共用）

条件注入
--------
说话人嵌入通过 ``speaker_proj`` 投影到隐藏空间，作为全局偏置注入。

输入序列构造
------------
::

    [ref_semantic_tokens, text_phonemes, <predict_start>, gen_semantic_tokens]

推理时只在 ``<predict_start>`` 之后自回归生成。

显存需求
--------
* ``float16`` 精度：约 2-4 GB GPU 显存
* ``float32`` 精度：约 4-8 GB GPU 显存
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from mosaic.nodes.audio.tts_backends.acoustic_models.base import AcousticModel

__all__ = ["GPT2ARModel"]


class GPT2ARModel(AcousticModel):
    """GPT-SoVITS GPT-2 自回归声学模型。

    使用 :class:`transformers.GPT2LMHeadModel` 作为骨干网络，配合双路径
    Embedding（文本 + 语义）和说话人投影层，将文本音素 token ids 自回归地
    转换为语义 token ids。

    Attributes
    ----------
    model_type : str
        固定为 ``"ar"``。
    vocab_size : int
        文本音素词表大小。
    semantic_vocab_size : int
        语义 token 词表大小。
    hidden_size : int
        隐藏层维度。
    """

    model_type: str = "ar"

    # 说话人嵌入条件强度（GPT-SoVITS 语义 token 模型中，将说话人嵌入以
    # 固定比例叠加到输入嵌入；见 E2-1：原为散落各处的魔法数字 0.1）。
    SPK_COND_SCALE: float = 0.1

    def __init__(
        self,
        model_path: str,
        vocab_size: int = 0,
        semantic_vocab_size: int = 1024,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        max_position_embeddings: int = 2048,
        num_speaker_embeddings: int = 512,
    ) -> None:
        """初始化 GPT2ARModel。

        Parameters
        ----------
        model_path : str
            GPT-SoVITS GPT 部分权重路径。
        vocab_size : int
            文本音素词表大小。
        semantic_vocab_size : int
            语义 token 词表大小。
        hidden_size : int
            GPT-2 隐藏维度。
        num_heads : int
            注意力头数。
        num_layers : int
            Transformer 层数。
        max_position_embeddings : int
            最大位置编码长度。
        num_speaker_embeddings : int
            说话人嵌入维度。
        """
        self._model_path: str = model_path
        self._semantic_vocab_size: int = semantic_vocab_size
        self._num_heads: int = num_heads
        self._num_layers: int = num_layers
        self._max_position_embeddings: int = max_position_embeddings
        self._num_speaker_embeddings: int = num_speaker_embeddings

        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size

        # 模型实例（load_weights 后填充）
        self._model: Any = None
        self._text_embedding: Any = None
        self._semantic_embedding: Any = None
        self._speaker_proj: Any = None
        self._device: str = "cpu"
        self._dtype: str = "float16"
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # 静态采样工具
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_dtype(dtype: str) -> Any:
        """将 dtype 字符串解析为 torch dtype。"""
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
        """nucleus sampling。"""
        import torch

        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
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
        """对已生成 token 的 logits 除以 penalty。"""
        import torch

        if penalty == 1.0:
            return logits
        for batch_idx in range(logits.size(0)):
            ids = generated_ids[batch_idx].unique().long()
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

    @staticmethod
    def stop_condition(
        current_token: Any,
        recent_tokens: list[Any],
        eos_token_id: int | None = None,
        max_repeat: int = 5,
    ) -> bool:
        """检查停止条件。

        .. note::
            本方法返回单一 ``bool``，仅在 ``batch_size=1`` 时能精确表达
            “该序列应停止”。对 ``batch_size > 1`` 的输入，使用 ``torch.all``
            聚合判断——只有当批次内所有序列同时满足停止条件时才返回 ``True``，
            从而避免对多元素张量调用 ``.item()`` 引发的 ``RuntimeError``
            （见 E4-4）。

        Parameters
        ----------
        current_token : Any
            当前 token。
        recent_tokens : list
            最近生成的 token 列表。
        eos_token_id : int | None
            EOS token id。
        max_repeat : int
            连续相同 token 的最大允许次数。

        Returns
        -------
        bool
            ``True`` 表示应停止生成。
        """
        import torch

        def _matches(token: Any, value: Any) -> bool:
            """token 与 value 是否全部相等（兼容多元素 tensor 与标量）。

            对 tensor 使用 ``torch.all`` 聚合，避免 ``.item()`` 在
            ``batch_size > 1`` 时抛 ``RuntimeError``。
            """
            eq = token == value
            try:
                # tensor（含多元素）→ torch.all 聚合；Python 标量 bool
                # 也可被 torch.all 处理为 0-dim tensor。
                return bool(torch.all(eq))
            except (TypeError, RuntimeError, ValueError):
                return bool(eq)

        # EOS 检测
        if eos_token_id is not None:
            if _matches(current_token, eos_token_id):
                return True

        # 重复检测
        if len(recent_tokens) >= max_repeat:
            if all(
                _matches(current_token, t) for t in recent_tokens[-max_repeat:]
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # 权重加载
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """加载模型权重。

        Parameters
        ----------
        weights_path : str
            权重目录路径。
        device : str
            目标设备。
        dtype : str
            数据精度。
        """
        import torch

        torch_dtype = self._parse_dtype(dtype)

        from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore

        # 创建 GPT2Config
        config_path = os.path.join(weights_path, "config.json")
        if os.path.exists(config_path):
            config = GPT2Config.from_pretrained(weights_path)
            config.vocab_size = self._semantic_vocab_size
        else:
            config = GPT2Config(
                vocab_size=self._semantic_vocab_size,
                n_positions=self._max_position_embeddings,
                n_embd=self.hidden_size,
                n_layer=self._num_layers,
                n_head=self._num_heads,
            )

        model = GPT2LMHeadModel(config)

        # 创建双路径 Embedding
        self._text_embedding = torch.nn.Embedding(
            self.vocab_size, self.hidden_size
        )
        self._semantic_embedding = torch.nn.Embedding(
            self._semantic_vocab_size, self.hidden_size
        )
        self._speaker_proj = torch.nn.Linear(
            self._num_speaker_embeddings, self.hidden_size
        )

        # 加载权重
        safetensors_path = os.path.join(
            weights_path, "acoustic_model.safetensors"
        )
        pytorch_path = os.path.join(weights_path, "acoustic_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file  # type: ignore

            state_dict = load_file(safetensors_path)
            model.load_state_dict(state_dict, strict=False)

            # 加载 Embedding 权重
            for key, val in state_dict.items():
                if "text_embedding" in key:
                    self._text_embedding.load_state_dict(
                        {"weight": val}, strict=False
                    )
                elif "semantic_embedding" in key or "hz_embedding" in key:
                    self._semantic_embedding.load_state_dict(
                        {"weight": val}, strict=False
                    )
                elif "speaker_proj" in key or "spk_proj" in key:
                    self._speaker_proj.load_state_dict(
                        {"weight": val}, strict=False
                    )
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=False)

        # 移动到 device/dtype
        model = model.to(device=device, dtype=torch_dtype)
        self._text_embedding = self._text_embedding.to(
            device=device, dtype=torch_dtype
        )
        self._semantic_embedding = self._semantic_embedding.to(
            device=device, dtype=torch_dtype
        )
        self._speaker_proj = self._speaker_proj.to(
            device=device, dtype=torch_dtype
        )

        model.eval()
        self._text_embedding.eval()
        self._semantic_embedding.eval()
        self._speaker_proj.eval()

        self._model = model
        self._device = device
        self._dtype = dtype
        self._is_loaded = True

    def unload_weights(self) -> None:
        """释放模型权重。"""
        self._model = None
        self._text_embedding = None
        self._semantic_embedding = None
        self._speaker_proj = None

        try:
            import torch

            from mosaic.core._device_utils import empty_device_cache

            empty_device_cache()
        except ImportError:
            pass

        self._is_loaded = False

    # ------------------------------------------------------------------
    # 访问器
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> Any:
        """返回文本音素 Embedding 层。"""
        return self._text_embedding

    def get_output_head(self) -> Any:
        """返回输出头。"""
        if self._model is not None:
            return self._model.lm_head
        return None

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
        """自回归生成语义 token。

        Parameters
        ----------
        token_ids : torch.Tensor
            文本音素 token ids，shape ``[1, text_len]``。
        speaker_embedding : Any | None
            包含 ``ref_semantic_tokens`` 和 ``speaker_embedding`` 的字典。
        max_new_tokens : int
            最大生成 token 数。
        temperature : float
            采样温度。
        top_p : float
            nucleus sampling 参数。
        top_k : int
            top-k 参数。

        Returns
        -------
        torch.Tensor
            生成的语义 token ids，shape ``[1, gen_len]``。
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call load_weights() before generate()."
            )

        import torch

        repetition_penalty: float = kwargs.get("repetition_penalty", 1.3)
        eos_token_id: int | None = kwargs.get("eos_token_id", None)
        max_repeat: int = kwargs.get("max_repeat", 5)

        device = self._device

        # 提取参考信息
        ref_tokens = None
        spk_emb = None
        if speaker_embedding is not None:
            if isinstance(speaker_embedding, dict):
                ref_tokens = speaker_embedding.get("ref_semantic_tokens")
                spk_emb = speaker_embedding.get("speaker_embedding")
            else:
                ref_tokens = speaker_embedding

        # 构造输入嵌入
        text_embeds = self._text_embedding(token_ids)

        input_embeds = text_embeds
        if ref_tokens is not None:
            ref_embeds = self._semantic_embedding(ref_tokens)
            input_embeds = torch.cat([ref_embeds, text_embeds], dim=1)

        # 说话人条件注入
        if spk_emb is not None:
            spk_cond = self._speaker_proj(spk_emb)
            if spk_cond.dim() == 1:
                spk_cond = spk_cond.unsqueeze(0).unsqueeze(0)
            elif spk_cond.dim() == 2:
                spk_cond = spk_cond.unsqueeze(1)
            # 广播到序列长度
            spk_cond = spk_cond.expand(
                -1, input_embeds.size(1), -1
            )
            input_embeds = input_embeds + spk_cond * self.SPK_COND_SCALE

        # 自回归生成
        generated_tokens: list[torch.Tensor] = []
        past_key_values: Any = None
        cur_embeds = input_embeds
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

            if temperature != 1.0:
                next_logits = next_logits / temperature

            next_logits = self._top_k_filtering(next_logits, top_k)
            next_logits = self._top_p_filtering(next_logits, top_p)

            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                gen_ids = torch.cat(generated_tokens, dim=-1)
                next_logits = self._apply_repetition_penalty(
                    next_logits, gen_ids, repetition_penalty
                )

            probs = torch.softmax(next_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(sampled)

            # 检查停止条件
            if self.stop_condition(
                sampled, generated_tokens, eos_token_id, max_repeat
            ):
                finished = True

            # 下一步输入
            next_embed = self._semantic_embedding(sampled)
            cur_embeds = next_embed

        if not generated_tokens:
            return torch.empty((1, 0), dtype=torch.long, device=device)

        result = torch.cat(generated_tokens, dim=-1)
        return result

    def generate_stream(
        self,
        token_ids: Any,
        speaker_embedding: Any | None = None,
        stream_batch: int = 16,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """流式生成语义 token。

        Parameters
        ----------
        token_ids : torch.Tensor
            文本音素 token ids。
        speaker_embedding : Any | None
            说话人信息字典。
        stream_batch : int
            每次 yield 的 token 数。

        Yields
        ------
        torch.Tensor
            增量的语义 token ids。
        """
        if not self._is_loaded or self._model is None:
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
        max_repeat: int = kwargs.get("max_repeat", 5)

        device = self._device

        ref_tokens = None
        spk_emb = None
        if speaker_embedding is not None:
            if isinstance(speaker_embedding, dict):
                ref_tokens = speaker_embedding.get("ref_semantic_tokens")
                spk_emb = speaker_embedding.get("speaker_embedding")
            else:
                ref_tokens = speaker_embedding

        text_embeds = self._text_embedding(token_ids)
        input_embeds = text_embeds
        if ref_tokens is not None:
            ref_embeds = self._semantic_embedding(ref_tokens)
            input_embeds = torch.cat([ref_embeds, text_embeds], dim=1)

        if spk_emb is not None:
            spk_cond = self._speaker_proj(spk_emb)
            if spk_cond.dim() == 1:
                spk_cond = spk_cond.unsqueeze(0).unsqueeze(0)
            elif spk_cond.dim() == 2:
                spk_cond = spk_cond.unsqueeze(1)
            spk_cond = spk_cond.expand(-1, input_embeds.size(1), -1)
            input_embeds = input_embeds + spk_cond * self.SPK_COND_SCALE

        buffer: list[torch.Tensor] = []
        past_key_values: Any = None
        cur_embeds = input_embeds
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

            if temperature != 1.0:
                next_logits = next_logits / temperature

            next_logits = self._top_k_filtering(next_logits, top_k)
            next_logits = self._top_p_filtering(next_logits, top_p)

            if repetition_penalty != 1.0 and len(buffer) > 0:
                gen_ids = torch.cat(buffer, dim=-1)
                next_logits = self._apply_repetition_penalty(
                    next_logits, gen_ids, repetition_penalty
                )

            probs = torch.softmax(next_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            buffer.append(sampled)

            if self.stop_condition(
                sampled, buffer, eos_token_id, max_repeat
            ):
                finished = True

            next_embed = self._semantic_embedding(sampled)
            cur_embeds = next_embed

            if len(buffer) >= stream_batch:
                chunk = torch.cat(buffer, dim=-1)
                yield chunk
                buffer = []

        if buffer:
            chunk = torch.cat(buffer, dim=-1)
            yield chunk
