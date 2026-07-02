# mosaic/nodes/audio/tts_backends/implementations/chattts_backend.py
"""ChatTTS 后端实现。

将 ChatTTS 的四层组件组装为统一的 :class:`TTSBackend`，提供阻塞合成与
流式合成能力。ChatTTS 采用「文本 → Llama 自回归声学模型 → DVAE 解码 →
Vocos 声码器」的管线，输出 24kHz 单声道波形。

四层组装
--------
* Layer 1 — :class:`ChatTokenizer`：文本清洗、韵律标记、分词、说话人嵌入解码。
* Layer 2 — :class:`LlamaARModel`：基于 Llama 的自回归声学模型，文本 token
  → 多组 VQ 音频码 token。
* Layer 3 — :class:`_CompositeVocoder`（DVAE + Vocos 复合声码器）：VQ token
  → mel（DVAE）→ waveform（Vocos）。
* Layer 4 — :class:`StreamAdapter`：流式缓冲与 chunk 切分。

设计要点
--------
* ``torch`` / ``transformers`` / ``safetensors`` 等重依赖采用惰性导入，使本
  模块在未安装这些依赖时仍可被导入与注册（仅在实际 ``load`` 时才报依赖缺失）。
* 四层组件（``ChatTokenizer`` / ``LlamaARModel`` / ``DVAEDecoder`` /
  ``VocosVocoder`` / ``StreamAdapter``）均在 :meth:`_build_pipeline` 内部
  延迟导入，避免模块加载阶段产生硬依赖，亦兼容各组件由其他子任务并行创建的
  情形。
* DVAE 与 Vocos 组合为 :class:`_CompositeVocoder` 以符合 :class:`TTSBackend`
  的四层架构（``self._vocoder.decode`` 接收 GPT 输出的 VQ token ids）。
* ``token_ids`` / ``speaker_embedding`` 等参数类型用 :data:`~typing.Any`
  标注，避免在模块顶层硬依赖 ``torch``。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from mosaic.core.types import AudioData
from mosaic.nodes.audio.tts_backends.base import TTSBackend, TTSBackendSpec
from mosaic.nodes.audio.tts_backends.vocoders.base import Vocoder

__all__ = ["ChatTTSBackend"]


# ---------------------------------------------------------------------------
# 复合声码器：DVAE(token→mel) + Vocos(mel→waveform)
# ---------------------------------------------------------------------------
class _CompositeVocoder(Vocoder):
    """复合声码器：DVAE(token→mel) + Vocos(mel→waveform)。

    ChatTTS 的声码器由两阶段组成：DVAE 将 GPT 输出的多组 VQ 音频码 token 解码
    为 mel 频谱，Vocos 再将 mel 频谱转换为波形。本类将二者组合为单个
    :class:`Vocoder` 子类，以符合 :class:`TTSBackend` 四层架构中 ``Layer 3``
    的单一声码器接口（``self._vocoder.decode(features)``）。

    权重不在本类中加载——由 :meth:`ChatTTSBackend._build_pipeline` 分别为 DVAE
    与 Vocos 调用 ``load_weights`` 后注入本类。

    Attributes
    ----------
    vocoder_type : str
        声码器类型，固定为 ``"vocos"``。
    input_type : str
        输入特征类型，固定为 ``"vq_tokens"``（GPT 输出的 VQ token ids）。
    sample_rate : int
        输出采样率，固定为 ``24000``。
    """

    vocoder_type: str = "vocos"
    input_type: str = "vq_tokens"
    sample_rate: int = 24000

    def __init__(self, dvae: Any, vocos: Any) -> None:
        """初始化复合声码器。

        Parameters
        ----------
        dvae : DVAEDecoder
            已加载权重的 DVAE 解码器实例。
        vocos : VocosVocoder
            已加载权重的 Vocos 声码器实例。
        """
        self._dvae: Any = dvae
        self._vocos: Any = vocos

    # ------------------------------------------------------------------
    # 权重管理（权重已在 _build_pipeline 中加载，此处为空实现）
    # ------------------------------------------------------------------
    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """权重加载（空实现）。

        DVAE 与 Vocos 的权重已在 :meth:`ChatTTSBackend._build_pipeline` 中
        分别加载并注入本类，故此处无需重复加载。
        """
        pass  # 权重已在 _build_pipeline 中加载

    def unload_weights(self) -> None:
        """释放 DVAE 与 Vocos 的权重。"""
        for comp in (self._dvae, self._vocos):
            unload = getattr(comp, "unload_weights", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # 解码
    # ------------------------------------------------------------------
    def decode(self, features: Any) -> tuple[Any, int]:
        """阻塞解码：VQ token ids → mel → waveform。

        Parameters
        ----------
        features : torch.Tensor
            GPT 输出的音频码 token ids，形状 ``[num_vq, frames]``。

        Returns
        -------
        tuple
            ``(waveform, sample_rate)``。
        """
        # features 是 GPT 输出的 token ids [num_vq, frames]
        mel = self._dvae.forward(features)
        waveform, sr = self._vocos.decode(mel)
        return waveform, sr

    def decode_chunk(self, features: Any) -> tuple[Any, int]:
        """流式解码：VQ token ids 增量块 → mel → waveform。

        Parameters
        ----------
        features : torch.Tensor
            GPT 流式输出的增量音频码 token ids，形状 ``[num_vq, chunk]``。

        Returns
        -------
        tuple
            ``(waveform, sample_rate)``。
        """
        mel = self._dvae.forward_chunk(features)
        waveform, sr = self._vocos.decode_chunk(mel)
        return waveform, sr


# ---------------------------------------------------------------------------
# ChatTTSBackend
# ---------------------------------------------------------------------------
class ChatTTSBackend(TTSBackend):
    """ChatTTS TTS 后端。

    将 ChatTTS 的文本前端（:class:`ChatTokenizer`）、声学模型
    （:class:`LlamaARModel`）、复合声码器（DVAE + Vocos）与流式适配器
    （:class:`StreamAdapter`）组装为统一的 :class:`TTSBackend`，支持中英文
    阻塞合成与流式合成，并通过随机种子生成说话人嵌入。

    生命周期
    --------
    1. 构造后端实例（``is_loaded=False``）。
    2. 调用 :meth:`load` 加载四层管线（``is_loaded=True``）。
    3. 调用 :meth:`synthesize` / :meth:`synthesize_stream` 合成语音。
    4. 调用 :meth:`unload` 释放资源。

    Examples
    --------
    >>> backend = ChatTTSBackend(model_path="/data/chattts")
    >>> backend.load(device="cuda", dtype="float16")
    >>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

    Notes
    -----
    ChatTTS 模型遵循 **CC BY-NC 4.0** 许可，仅供非商业用途。
    """

    # ==================================================================
    # 类属性：后端规格
    # ==================================================================
    name: str = "chattts"
    spec: TTSBackendSpec = TTSBackendSpec(
        name="chattts",
        supported_languages=["zh", "en"],
        supports_streaming=True,
        supports_voice_clone=True,
        vocoder_type="vocos",
        acoustic_type="ar",
        min_gpu_memory_gb=2.0,
        model_license="CC BY-NC 4.0",
        sample_rate=24000,
        default_params={
            "temperature": 0.3,
            "top_p": 0.7,
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
    )

    # 常用说话人种子列表（ChatTTS 通过种子随机生成说话人嵌入）
    _COMMON_SEEDS: list[int] = [2, 222, 786, 2024, 6653, 7114]

    # 默认说话人嵌入维度（与模型隐藏层维度匹配；spk_stat 存在时以实际值为准）
    _SPK_EMBED_DIM: int = 256

    # ==================================================================
    # 构造函数
    # ==================================================================
    def __init__(
        self,
        model_path: str,
        vocos_path: str | None = None,
        num_vq: int = 4,
        language: str = "zh",
        use_flash_attention: bool = True,
        streaming_enabled: bool = True,
        stream_batch: int = 24,
        scheduler: Any = None,
        repo_id: str | None = None,
    ) -> None:
        """初始化 ChatTTS 后端。

        Parameters
        ----------
        model_path : str
            ChatTTS 模型目录路径。若本地目录不存在或为空，将通过
            :class:`HFModelManager` 从 HuggingFace 仓库下载（默认
            ``2Noise/ChatTTS``）。HF 仓库布局为 ``config/`` (YAML 配置)
            + ``asset/`` (权重与 tokenizer)。
        vocos_path : str | None, default None
            Vocos 权重路径；``None`` 时自动从 ``asset/Vocos.safetensors``
            查找。提供自定义路径时优先使用。
        num_vq : int, default 4
            VQ 码本组数。若 ``config/gpt.yaml`` 存在，以配置值为准。
        language : str, default "zh"
            默认语言代码。
        use_flash_attention : bool, default True
            声学模型是否使用 Flash Attention 加速。
        streaming_enabled : bool, default True
            是否启用流式合成（构建 Layer 4 流式适配器）。
        scheduler : Any
            显存调度器实例，``None`` 使用全局单例。透传给
            :meth:`TTSBackend.__init__`。
        repo_id : str | None, default None
            HuggingFace 仓库 ID（如 ``"2Noise/ChatTTS"``）。``None`` 时
            使用 :attr:`HFModelManager.DEFAULT_REPOS` 中 ``"chattts"`` 对应
            的默认仓库。
        """
        super().__init__(scheduler=scheduler)

        # 构造参数
        self._model_path: str = model_path
        self._vocos_path: str | None = vocos_path
        self._num_vq: int = num_vq
        self._language: str = language
        self._use_flash_attention: bool = use_flash_attention
        self._streaming_enabled: bool = streaming_enabled
        # D2-2: 流式生成每次 yield 的 token 数（原硬编码 24）
        self._stream_batch: int = stream_batch
        self._repo_id: str | None = repo_id

        # 解析后的模型目录（_build_pipeline 中由 HFModelManager 填充）
        self._model_dir: str = model_path
        # Embed 权重路径（_build_pipeline 中填充，供说话人嵌入使用）
        self._embed_path: str = ""

        # DVAE / Vocos 实例引用（_build_pipeline 中填充，便于异常清理）
        self._dvae: Any = None
        self._vocos: Any = None

    # ==================================================================
    # 生命周期：组装 / 销毁管线
    # ==================================================================
    def _build_pipeline(self) -> None:
        """组装四层管线。

        依次构建并加载：

        1. Layer 1 — :class:`ChatTokenizer`（文本前端）
        2. Layer 2 — :class:`LlamaARModel`（声学模型）
        3. Layer 3a — :class:`DVAEDecoder`（VQ token → mel）
        4. Layer 3b — :class:`VocosVocoder`（mel → waveform）
        5. Layer 3  — :class:`_CompositeVocoder`（组合 3a + 3b）
        6. Layer 4 — :class:`StreamAdapter`（流式适配，按需构建）

        路径解析遵循 HuggingFace 仓库 ``2Noise/ChatTTS`` 的实际布局：
        ``config/`` (YAML 配置) + ``asset/`` (权重与 tokenizer)。
        通过 :class:`HFModelManager` 统一查找文件，兼容多种文件命名
        (``.safetensors`` / ``.pt``) 与目录结构。

        所有组件均在此方法内部延迟导入，避免模块加载阶段的硬依赖。
        """
        # 延迟导入四层组件与 HF 模型管理器，避免模块加载阶段产生硬依赖
        from mosaic.nodes.audio.tts_backends.acoustic_models.llama_ar import (
            LlamaARModel,
        )
        from mosaic.nodes.audio.tts_backends.hf_model_manager import HFModelManager
        from mosaic.nodes.audio.tts_backends.streaming.base import StreamAdapter
        from mosaic.nodes.audio.tts_backends.text_frontends.chat_tokenizer import (
            ChatTokenizer,
        )
        from mosaic.nodes.audio.tts_backends.vocoders.dvae import DVAEDecoder
        from mosaic.nodes.audio.tts_backends.vocoders.vocos import VocosVocoder

        # ------------------------------------------------------------------
        # 确保 HF 模型已下载
        # ------------------------------------------------------------------
        # 本地目录已存在时（含空目录，兼容测试与自定义布局）直接使用，
        # 不触发下载；目录不存在时通过 HFModelManager 从 HF 下载。
        # 下载失败（离线/无网络）时回退到本地路径，由后续路径解析兜底。
        if os.path.isdir(self._model_path):
            model_dir = self._model_path
        else:
            try:
                model_dir = HFModelManager.ensure_model(
                    self._model_path,
                    repo_id=self._repo_id,
                    backend_name="chattts",
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to ensure HF model download (%s); "
                    "falling back to local path %s",
                    exc,
                    self._model_path,
                )
                model_dir = self._model_path
        self._model_dir = model_dir

        # ------------------------------------------------------------------
        # 读取 YAML 配置获取模型超参数（覆盖默认值）
        # ------------------------------------------------------------------
        gpt_config = HFModelManager.load_yaml_config(
            os.path.join(model_dir, "config", "gpt.yaml")
        )
        num_audio_tokens: int = gpt_config.get("num_audio_tokens", 626)
        num_text_tokens: int = gpt_config.get("num_text_tokens", 21178)
        num_vq: int = gpt_config.get("num_vq", self._num_vq)
        # 同步 num_vq 到实例，确保各组件使用一致值
        self._num_vq = num_vq

        vocos_config = HFModelManager.load_yaml_config(
            os.path.join(model_dir, "config", "vocos.yaml")
        )
        vocos_sample_rate: int = vocos_config.get(
            "sample_rate", self.spec.sample_rate
        )
        vocos_n_mels: int = vocos_config.get("n_mels", 100)

        self._logger.debug(
            "ChatTTS config: num_audio_tokens=%d num_text_tokens=%d "
            "num_vq=%d vocos_sample_rate=%d vocos_n_mels=%d",
            num_audio_tokens,
            num_text_tokens,
            num_vq,
            vocos_sample_rate,
            vocos_n_mels,
        )

        # ------------------------------------------------------------------
        # Layer 1: 文本前端 —— ChatTokenizer
        # ------------------------------------------------------------------
        # Tokenizer 路径解析（按优先级）：
        #   1. asset/tokenizer/ 目录 (HuggingFace tokenizer 格式)
        #   2. asset/tokenizer.pt (PyTorch tokenizer)
        #   3. vocab.json (兼容旧布局)
        #   4. 空字符串 → 字符级分词
        tokenizer_dir = HFModelManager.find_dir(model_dir, ["asset/tokenizer"])
        tokenizer_pt = HFModelManager.find_file(
            model_dir, ["asset/tokenizer.pt"]
        )
        vocab_json = HFModelManager.find_file(
            model_dir, ["vocab.json", "asset/vocab.json"]
        )

        if tokenizer_dir:
            vocab_path = tokenizer_dir
        elif tokenizer_pt:
            vocab_path = tokenizer_pt
        elif vocab_json:
            vocab_path = vocab_json
        else:
            vocab_path = ""  # 字符级分词

        self._logger.debug(
            "ChatTTS tokenizer path: %s", vocab_path or "(char-level)"
        )
        self._text_frontend = ChatTokenizer(
            vocab_path=vocab_path,
            num_vq=self._num_vq,
            sample_rate=self.spec.sample_rate,
            num_text_tokens=num_text_tokens,
        )

        # ------------------------------------------------------------------
        # Layer 2: 声学模型 —— LlamaARModel
        # ------------------------------------------------------------------
        # GPT model 路径解析（按优先级）：
        #   1. asset/gpt/ 目录 (config.json + model.safetensors, transformers 格式)
        #   2. asset/GPT.pt (PyTorch checkpoint)
        gpt_dir = HFModelManager.find_dir(model_dir, ["asset/gpt"])
        gpt_pt = HFModelManager.find_weight(
            model_dir, "GPT", subdirs=["asset", ""]
        )

        if gpt_dir:
            gpt_model_path = gpt_dir
            gpt_weights_path = gpt_dir
        elif gpt_pt:
            gpt_model_path = model_dir
            gpt_weights_path = gpt_pt
        else:
            gpt_model_path = model_dir
            gpt_weights_path = model_dir

        self._logger.debug(
            "ChatTTS GPT model path: %s | weights: %s",
            gpt_model_path,
            gpt_weights_path,
        )
        self._acoustic_model = LlamaARModel(
            model_path=gpt_model_path,
            num_vq=self._num_vq,
            num_audio_tokens=num_audio_tokens,
            num_text_tokens=num_text_tokens,
            use_flash_attention=self._use_flash_attention,
        )
        self._acoustic_model.load_weights(
            weights_path=gpt_weights_path,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # DVAE 解码器 —— VQ token → mel
        # ------------------------------------------------------------------
        # ChatTTS 官方默认 use_decoder=True，使用 Decoder.safetensors（独立的
        # 解码器 DVAE，结构更优）。DVAE.safetensors 用于编码音频提取 speaker。
        # 优先 Decoder，回退到 DVAE。
        dvae_path = HFModelManager.find_weight(
            model_dir, "Decoder", subdirs=["asset", ""]
        )
        if not dvae_path:
            dvae_path = HFModelManager.find_weight(
                model_dir, "DVAE", subdirs=["asset", ""]
            )

        self._logger.debug(
            "ChatTTS DVAE weights path: %s", dvae_path or "(not found)"
        )
        self._dvae = DVAEDecoder(
            num_vq=self._num_vq,
            num_audio_tokens=num_audio_tokens,
            hidden_size=512,
            mel_bins=vocos_n_mels,
        )
        self._dvae.load_weights(
            weights_path=dvae_path,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # Layer 3b: Vocos 声码器 —— mel → waveform
        # ------------------------------------------------------------------
        # Vocos 权重：优先用户自定义路径 (vocos_path)，否则从 HF 布局查找
        if self._vocos_path:
            vocos_path = self._vocos_path
        else:
            vocos_path = HFModelManager.find_weight(
                model_dir, "Vocos", subdirs=["asset", ""]
            )

        self._logger.debug(
            "ChatTTS Vocos weights path: %s", vocos_path or "(not found)"
        )
        self._vocos = VocosVocoder(
            model_path=vocos_path,
            n_mels=vocos_n_mels,
            sample_rate=vocos_sample_rate,
        )
        self._vocos.load_weights(
            weights_path=vocos_path,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # Layer 3: 复合声码器 —— 组合 DVAE + Vocos
        # ------------------------------------------------------------------
        self._vocoder = _CompositeVocoder(self._dvae, self._vocos)

        # ------------------------------------------------------------------
        # Embed 权重路径（说话人嵌入，供 sample_random_speaker 等使用）
        # ------------------------------------------------------------------
        self._embed_path = HFModelManager.find_weight(
            model_dir, "Embed", subdirs=["asset", ""]
        )
        self._logger.debug(
            "ChatTTS Embed weights path: %s", self._embed_path or "(not found)"
        )

        # ------------------------------------------------------------------
        # Layer 4: 流式适配器 —— 按需构建
        # ------------------------------------------------------------------
        if self._streaming_enabled:
            self._stream_adapter = StreamAdapter(
                chunk_size=4096,
                overlap=256,
                sample_rate=self.spec.sample_rate,
            )

    def _destroy_pipeline(self) -> None:
        """销毁四层管线并释放资源。

        先调用基类实现（卸载 ``_vocoder`` 复合声码器，其内部会卸载 DVAE 与
        Vocos），再兜底清理 ``_dvae`` / ``_vocos`` 引用（覆盖管线组装中途失败、
        复合声码器尚未创建的情形）。
        """
        # 基类实现：依次卸载 text_frontend / acoustic_model / vocoder / stream_adapter
        super()._destroy_pipeline()

        # 兜底：若 _build_pipeline 中途失败，dvae / vocos 可能已加载但尚未被
        # 复合声码器持有；此处单独释放（unload_weights 幂等，重复调用安全）。
        for name, comp in (("dvae", self._dvae), ("vocos", self._vocos)):
            if comp is None:
                continue
            unload = getattr(comp, "unload_weights", None)
            if callable(unload):
                try:
                    unload()
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug("Error unloading %s weights: %s", name, exc)
        self._dvae = None
        self._vocos = None

    # ==================================================================
    # 核心合成
    # ==================================================================
    def synthesize(
        self,
        text: str,
        speaker: str | None = None,
        language: str = "zh",
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AudioData:
        """阻塞式合成完整语音。

        完整流程：

        1. 检查 ``is_loaded``。
        2. 文本校验。
        3. 提取韵律提示 ``prosody_prompt``（从 ``kwargs`` 中移除，避免透传到
           声学模型）。
        4. 文本清洗（``ChatTokenizer.preprocess``）。
        5. 韵律标记插入（``ChatTokenizer.insert_prosody_tokens``）。
        6. 分词（``ChatTokenizer.tokenize``）→ token_ids。
        7. 说话人嵌入解码（``ChatTokenizer.encode_speaker``）。
        8. 合并推理参数（``spec.default_params`` 与 ``kwargs``）。
        9. 声学模型生成（``LlamaARModel.generate``）→ audio_codes。
        10. 复合声码器解码（``_CompositeVocoder.decode``）→ waveform。
        11. 构造 :class:`AudioData` 返回。

        .. note::
           步骤 4-6 由 ``ChatTokenizer.tokenize`` 内部统一完成（清洗在前、
           韵律在后），此处透传 ``prosody_prompt`` 与 ``speaker_id`` 由其
           统一处理，以保证韵律特殊标记不会被重复预处理破坏。

        Parameters
        ----------
        text : str
            待合成文本。
        speaker : str | None
            说话人标识（Base16384 编码的说话人嵌入字符串）；``None`` 使用
            空说话人。
        language : str
            语言代码，默认 ``"zh"``。
        speed : float
            语速倍率（记录于元数据；ChatTTS 通过韵律标记控制语速）。
        **kwargs : Any
            额外参数，包括 ``prosody_prompt``（韵律提示）以及透传给声学模型
            的 ``temperature`` / ``top_p`` / ``top_k`` / ``max_new_tokens``
            等（覆盖 ``spec.default_params``）。

        Returns
        -------
        AudioData
            合成结果，``metadata`` 含 ``backend``/``text``/``speaker``
            /``language``/``speed``/``duration``/``sample_rate``/``streaming``。

        Raises
        ------
        RuntimeError
            后端未加载。
        ValueError
            文本为空。
        """
        # 1. 检查加载状态
        self._ensure_loaded()
        # 2. 文本校验
        if not isinstance(text, str) or not text.strip():
            raise ValueError("synthesize requires a non-empty 'text' string.")
        # A3-1: 统一 speaker 类型校验
        self._validate_speaker(speaker)

        self._logger.info(
            "synthesize: backend=%s language=%s speaker=%s speed=%.2f text_len=%d",
            self.name,
            language,
            speaker,
            speed,
            len(text),
        )

        # 3. 提取韵律提示（从 kwargs 中移除，避免透传到声学模型 generate）
        prosody_prompt: str = kwargs.pop("prosody_prompt", "")

        # 4-6. 文本前端：清洗 -> 韵律标记 -> 分词
        #     ChatTokenizer.tokenize 内部依次执行 preprocess、insert_prosody_tokens、
        #     _encode_text，此处透传 prosody_prompt 与 speaker_id 由其统一完成，
        #     保证韵律特殊标记（如 [break_4]）不会被重复预处理过滤。
        token_ids = self._text_frontend.tokenize(
            text,
            language=language,
            prosody_prompt=prosody_prompt,
            speaker_id=speaker,
        )

        # 7. 说话人嵌入解码
        speaker_embedding = self._text_frontend.encode_speaker(speaker)

        # 8. 合并推理参数（spec.default_params 为底，kwargs 覆盖）
        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        # 9. 声学模型生成 —— audio_codes: [num_vq, frames]
        audio_codes = self._acoustic_model.generate(
            token_ids,
            speaker_embedding,
            **params,
        )

        # 10. 复合声码器解码 —— VQ token -> mel -> waveform
        waveform, sample_rate = self._decode_full(audio_codes)

        # 11. 构造 AudioData 返回
        duration = self._compute_duration(waveform, sample_rate)
        metadata: dict[str, Any] = {
            "backend": self.name,
            "text": text,
            "speaker": speaker,
            "language": language,
            "speed": speed,
            "duration": duration,
            "sample_rate": sample_rate,
            "streaming": False,
        }
        return AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            metadata=metadata,
        )

    def synthesize_stream(
        self,
        text: str,
        speaker: str | None = None,
        language: str = "zh",
        speed: float = 1.0,
        chunk_size: int = 4096,
        **kwargs: Any,
    ) -> Iterator[AudioData]:
        """流式合成语音，逐块 yield :class:`AudioData`。

        流程：

        1. 前处理同 :meth:`synthesize`（步骤 1-8）。
        2. 创建 :class:`StreamSession` 流式会话。
        3. 对声学模型的流式输出逐块迭代：

           a. ``LlamaARModel.generate_stream`` → audio_codes_chunk
           b. 复合声码器 ``decode_chunk`` → waveform_chunk
           c. ``StreamSession.push(waveform_chunk)``
           d. ``StreamSession.pop()`` → yield :class:`AudioData`

        4. 冲刷缓冲区中剩余数据。

        若后端不支持流式（``streaming_enabled=False``），回退为
        :meth:`synthesize` 一次性返回完整结果。

        Parameters
        ----------
        text : str
            待合成文本。
        speaker : str | None
            说话人标识。
        language : str
            语言代码，默认 ``"zh"``。
        speed : float
            语速倍率。
        chunk_size : int
            每个音频块的目标采样数，默认 ``4096``。
        **kwargs : Any
            额外参数，同 :meth:`synthesize`。

        Yields
        ------
        AudioData
            逐块音频数据，``metadata`` 中 ``streaming=True``。
        """
        # 1. 检查加载状态
        self._ensure_loaded()
        # 2. 文本校验
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "synthesize_stream requires a non-empty 'text' string."
            )

        # 不支持流式：回退为一次性合成
        if not self.spec.supports_streaming or self._stream_adapter is None:
            self._logger.info(
                "Backend %r streaming disabled; "
                "falling back to blocking synthesize.",
                self.name,
            )
            yield self.synthesize(
                text,
                speaker=speaker,
                language=language,
                speed=speed,
                **kwargs,
            )
            return

        self._logger.info(
            "synthesize_stream: backend=%s language=%s speaker=%s "
            "speed=%.2f chunk_size=%d text_len=%d",
            self.name,
            language,
            speaker,
            speed,
            chunk_size,
            len(text),
        )

        # 3. 提取韵律提示
        prosody_prompt: str = kwargs.pop("prosody_prompt", "")

        # 4-6. 文本前端：清洗 -> 韵律标记 -> 分词
        token_ids = self._text_frontend.tokenize(
            text,
            language=language,
            prosody_prompt=prosody_prompt,
            speaker_id=speaker,
        )

        # 7. 说话人嵌入解码
        speaker_embedding = self._text_frontend.encode_speaker(speaker)

        # 8. 合并推理参数 + 流式批次大小
        params: dict[str, Any] = dict(self.spec.default_params)
        params["stream_batch"] = self._stream_batch  # D2-2: 可配置（kwargs 可覆盖）
        params.update(kwargs)

        # 创建流式会话并尝试配置块大小（Layer 4 预热）
        session = self._get_stream_session(chunk_size)

        try:
            # 9. 流式生成 -> 逐块解码 -> 缓冲输出
            for audio_codes_chunk in self._acoustic_model.generate_stream(
                token_ids,
                speaker_embedding,
                **params,
            ):
                # 流式取消：提前终止生成循环
                if session.is_cancelled is True:
                    self._logger.info(
                        "synthesize_stream cancelled for backend %s",
                        self.name,
                    )
                    break
                # 10. 复合声码器流式解码：VQ token 块 -> mel 块 -> waveform 块
                waveform_chunk, _sample_rate = self._decode_chunk(audio_codes_chunk)
                # 推入缓冲区
                self._stream_push(session, waveform_chunk)
                # 弹出已凑齐的块
                yield from self._stream_drain(
                    session, text, speaker, language, speed
                )

            # 11. 冲刷缓冲区中剩余数据
            yield from self._stream_finish(
                session, text, speaker, language, speed
            )
        finally:
            # 确保会话缓冲与声学模型 KV cache 被释放/重置（即使中途抛异常）
            self._cleanup_stream_state(session)

    # ==================================================================
    # 查询与说话人管理
    # ==================================================================
    def list_speakers(self) -> list[str]:
        """返回内置的说话人列表。

        ChatTTS 通过随机种子（Seed）生成说话人嵌入，本身不内置固定说话人。
        此处返回常用的种子标识列表，用户可结合 :meth:`set_seed` 与
        :meth:`sample_random_speaker` 生成对应的说话人嵌入字符串，再作为
        ``speaker`` 参数传入 :meth:`synthesize`。

        Returns
        -------
        list[str]
            常用种子标识列表，如 ``["seed_2", "seed_222", ...]``。
        """
        return [f"seed_{s}" for s in self._COMMON_SEEDS]

    def sample_random_speaker(self) -> str:
        """随机采样一个说话人嵌入并编码为字符串。

        采用高斯分布采样：``spk = randn * std + mean``，其中 ``mean`` / ``std``
        从模型路径下的 ``asset/spk_stat.pt`` 加载（若存在），否则使用默认值。
        采样后经 ``float16`` 量化、LZMA2 压缩、Base16384 编码为字符串，与
        :meth:`ChatTokenizer.encode_speaker` 的解码流程对称。

        Returns
        -------
        str
            Base16384 + LZMA2 编码的说话人嵌入字符串，可直接作为 ``speaker``
            参数传入 :meth:`synthesize`。

        Raises
        ------
        ImportError
            ``torch`` 未安装。
        """
        import lzma
        import struct

        import torch

        # 加载 mean / std（若 spk_stat.pt 存在）
        # spk_stat.pt 在 HF 布局中位于 asset/spk_stat.pt
        from mosaic.nodes.audio.tts_backends.hf_model_manager import HFModelManager

        dim = self._SPK_EMBED_DIM
        mean: Any = torch.zeros(dim)
        std: Any = torch.ones(dim)
        stat_path = HFModelManager.find_file(
            self._model_path, ["asset/spk_stat.pt", "spk_stat.pt"]
        )
        if stat_path:
            try:
                stat = torch.load(stat_path, map_location="cpu", weights_only=False)
                mean = stat.get("mean", mean)
                std = stat.get("std", std)
                dim = int(mean.shape[-1]) if hasattr(mean, "shape") else dim
            except Exception as exc:  # noqa: BLE001
                # E3-1: spk_stat.pt 加载失败影响说话人采样质量，应可见
                self._logger.warning(
                    "Failed to load spk_stat.pt: %s", exc, exc_info=True
                )

        # 高斯分布采样：spk = randn * std + mean
        spk = torch.randn(dim) * std + mean
        spk = spk.to(torch.float16)

        # 编码：float16 字节 -> LZMA2 压缩 -> Base16384 编码
        n = spk.numel()
        raw = struct.pack("<" + "e" * n, *spk.tolist())
        compressed = lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[
                {"id": lzma.FILTER_LZMA2, "preset": 7 | lzma.PRESET_EXTREME}
            ],
        )
        return self._base16384_encode(compressed)

    def set_seed(self, seed: int) -> None:
        """设置随机种子（影响说话人采样与声学模型生成）。

        Parameters
        ----------
        seed : int
            随机种子。
        """
        import random

        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        self._logger.debug("Random seed set to %d.", seed)

    # ==================================================================
    # 依赖检查
    # ==================================================================
    @classmethod
    def check_dependencies(cls) -> bool:
        """检查运行时依赖是否可用。

        检查 ``torch`` 与 ``transformers`` 是否已安装且可导入。

        Returns
        -------
        bool
            依赖齐全返回 ``True``，否则 ``False``。
        """
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    # ==================================================================
    # 内部辅助：Base16384 编码
    # ==================================================================
    @staticmethod
    def _base16384_encode(data: bytes) -> str:
        """将字节序列编码为 Base16384 字符串。

        与 :meth:`ChatTokenizer._base16384_decode` 对称：每 7 字节（56 bit）
        拆分为 4 个 14-bit 值，每个值映射为一个 Unicode 字符（码点 =
        ``0x4E00 + 值``）。剩余不足 7 字节按 14-bit 对齐编码。

        Parameters
        ----------
        data : bytes
            待编码的字节序列。

        Returns
        -------
        str
            Base16384 编码字符串。
        """
        OFFSET: int = 0x4E00
        chars: list[str] = []
        n = len(data)
        full = n - (n % 7)

        # 完整块：每 7 字节 -> 4 个 14-bit 值
        for i in range(0, full, 7):
            bits = int.from_bytes(data[i:i + 7], "big")
            chars.append(chr(OFFSET + ((bits >> 42) & 0x3FFF)))
            chars.append(chr(OFFSET + ((bits >> 28) & 0x3FFF)))
            chars.append(chr(OFFSET + ((bits >> 14) & 0x3FFF)))
            chars.append(chr(OFFSET + (bits & 0x3FFF)))

        # 剩余字节 -> 14-bit 值（左侧补零对齐）
        rem = data[full:]
        if rem:
            bits = int.from_bytes(rem, "big")
            nbits = 8 * len(rem)
            nvals = (nbits + 13) // 14  # 向上取整为 14-bit 值数量
            bits <<= (nvals * 14 - nbits)  # 左侧补零对齐到 14-bit 边界
            for j in range(nvals):
                shift = (nvals - 1 - j) * 14
                chars.append(chr(OFFSET + ((bits >> shift) & 0x3FFF)))
            # E5-2：追加余数长度标记 \u3D0r，使解码端能精确还原原始字节数
            chars.append(chr(0x3D00 + len(rem)))

        return "".join(chars)
