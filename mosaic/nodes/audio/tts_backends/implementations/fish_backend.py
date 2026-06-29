# mosaic/nodes/audio/tts_backends/implementations/fish_backend.py
"""Fish Speech TTS 后端实现。

文件路径: mosaic/nodes/audio/tts_backends/implementations/fish_backend.py

将 Fish Speech 的四层组件组装为统一的 :class:`TTSBackend`，提供阻塞合成与
流式合成能力。Fish Speech 采用「文本 → Llama 自回归声学模型 → VQ 解码 →
HiFi-GAN 声码器」的管线，输出 22050Hz 单声道波形。

四层组装
--------
* Layer 1 — :class:`FishTokenizer`：文本清洗、语言标记插入、字符级/BPE 分词、
  语音克隆 token 序列构造。
* Layer 2 — :class:`FishLlamaARModel`：基于 Llama 的自回归声学模型，统一
  词表 Embedding，文本 token → 音频 codec token。支持语音克隆（参考音频
  codec tokens 拼接到输入序列前部）。
* Layer 3 — :class:`_CompositeVocoder`（VQDecoder + HiFiGanVocoder 复合
  声码器）：codec token → mel（VQDecoder）→ waveform（HiFi-GAN）。
* Layer 4 — :class:`StreamAdapter`：流式缓冲与 chunk 切分。

与 ChatTTS 后端的差异
---------------------
* **采样率**：Fish Speech 使用 22050Hz（ChatTTS 使用 24000Hz）。
* **声码器**：VQDecoder + HiFi-GAN（ChatTTS 使用 DVAE + Vocos）。
* **语音克隆**：通过参考音频的 codec tokens 序列实现（ChatTTS 通过
  spk_emb 向量替换实现）。
* **语言支持**：中、英、日、韩四语言（ChatTTS 仅中英）。
* **许可证**：Apache-2.0（ChatTTS 为 CC BY-NC 4.0）。

设计要点
--------
* ``torch`` / ``transformers`` / ``safetensors`` 等重依赖采用惰性导入。
* 四层组件均在 :meth:`_build_pipeline` 内部延迟导入。
* VQDecoder 与 HiFiGanVocoder 组合为 :class:`_CompositeVocoder` 以符合
  :class:`TTSBackend` 四层架构。
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

__all__ = ["FishSpeechBackend"]


# ---------------------------------------------------------------------------
# 复合声码器：VQDecoder(codec token → mel) + HiFi-GAN(mel → waveform)
# ---------------------------------------------------------------------------
class _CompositeVocoder(Vocoder):
    """复合声码器：VQDecoder(codec token → mel) + HiFi-GAN(mel → waveform)。

    Fish Speech 的声码器由两阶段组成：VQDecoder 将 LLaMA 输出的 codec
    token 解码为 mel 频谱，HiFi-GAN 再将 mel 频谱转换为波形。本类将二者
    组合为单个 :class:`Vocoder` 子类，以符合 :class:`TTSBackend` 四层架构
    中 ``Layer 3`` 的单一声码器接口。

    权重不在本类中加载——由 :meth:`FishSpeechBackend._build_pipeline` 分别
    为 VQDecoder 与 HiFiGanVocoder 调用 ``load_weights`` 后注入本类。

    Attributes
    ----------
    vocoder_type : str
        声码器类型，固定为 ``"hifi_gan"``。
    input_type : str
        输入特征类型，固定为 ``"codec_tokens"``。
    sample_rate : int
        输出采样率，固定为 ``22050``。
    """

    vocoder_type: str = "hifi_gan"
    input_type: str = "codec_tokens"
    sample_rate: int = 22050

    def __init__(self, vq_decoder: Any, hifi_gan: Any) -> None:
        """初始化复合声码器。

        Parameters
        ----------
        vq_decoder : VQDecoder
            已加载权重的 VQ 解码器实例。
        hifi_gan : HiFiGanVocoder
            已加载权重的 HiFi-GAN 声码器实例。
        """
        self._vq_decoder: Any = vq_decoder
        self._hifi_gan: Any = hifi_gan

    def load_weights(
        self, weights_path: str, device: str = "cuda", dtype: str = "float16"
    ) -> None:
        """权重加载（空实现）。

        VQDecoder 与 HiFiGanVocoder 的权重已在
        :meth:`FishSpeechBackend._build_pipeline` 中分别加载并注入本类。
        """
        pass

    def unload_weights(self) -> None:
        """释放 VQDecoder 与 HiFiGanVocoder 的权重。"""
        for comp in (self._vq_decoder, self._hifi_gan):
            unload = getattr(comp, "unload_weights", None)
            if callable(unload):
                try:
                    unload()
                except Exception:  # noqa: BLE001
                    pass

    def decode(self, features: Any) -> tuple[Any, int]:
        """阻塞解码：codec token → mel → waveform。

        Parameters
        ----------
        features : torch.Tensor
            LLaMA 输出的音频 codec token ids，形状 ``[1, frames]`` 或
            ``[num_codebooks, frames]``。

        Returns
        -------
        tuple
            ``(waveform, sample_rate)``。
        """
        mel = self._vq_decoder.forward(features)
        waveform, sr = self._hifi_gan.decode(mel)
        return waveform, sr

    def decode_chunk(self, features: Any) -> tuple[Any, int]:
        """流式解码：codec token 增量块 → mel → waveform。

        Parameters
        ----------
        features : torch.Tensor
            LLaMA 流式输出的增量 codec token ids。

        Returns
        -------
        tuple
            ``(waveform, sample_rate)``。
        """
        mel = self._vq_decoder.forward_chunk(features)
        waveform, sr = self._hifi_gan.decode_chunk(mel)
        return waveform, sr


# ---------------------------------------------------------------------------
# FishSpeechBackend
# ---------------------------------------------------------------------------
class FishSpeechBackend(TTSBackend):
    """Fish Speech TTS 后端。

    将 Fish Speech 的文本前端（:class:`FishTokenizer`）、声学模型
    （:class:`FishLlamaARModel`）、复合声码器（VQDecoder + HiFi-GAN）与
    流式适配器（:class:`StreamAdapter`）组装为统一的 :class:`TTSBackend`，
    支持中英日韩四语言阻塞合成与流式合成，并通过参考音频实现语音克隆。

    生命周期
    --------
    1. 构造后端实例（``is_loaded=False``）。
    2. 调用 :meth:`load` 加载四层管线（``is_loaded=True``）。
    3. 调用 :meth:`synthesize` / :meth:`synthesize_stream` 合成语音。
    4. 调用 :meth:`unload` 释放资源。

    Examples
    --------
    >>> backend = FishSpeechBackend(model_path="/data/fish_speech")
    >>> backend.load(device="cuda", dtype="float16")
    >>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

    Notes
    -----
    Fish Speech 模型遵循 **Apache-2.0** 许可。输出采样率为 22050Hz。
    """

    name: str = "fish"
    spec: TTSBackendSpec = TTSBackendSpec(
        name="fish",
        supported_languages=["zh", "en", "ja", "ko"],
        supports_streaming=True,
        supports_voice_clone=True,
        vocoder_type="hifi_gan",
        acoustic_type="ar",
        min_gpu_memory_gb=3.0,
        model_license="Apache-2.0",
        sample_rate=22050,
        default_params={
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 200,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2048,
        },
    )

    # 内置音色列表（Fish Speech 主要通过参考音频实现任意音色）
    _BUILTIN_SPEAKERS: list[str] = ["default", "male", "female"]

    def __init__(
        self,
        model_path: str,
        hifi_gan_path: str | None = None,
        audio_encoder_path: str | None = None,
        codec_type: str = "dac",
        language: str = "zh",
        use_flash_attention: bool = True,
        streaming_enabled: bool = True,
        scheduler: Any = None,
        repo_id: str | None = None,
    ) -> None:
        """初始化 Fish Speech 后端。

        Parameters
        ----------
        model_path : str
            Fish Speech 模型权重路径。支持两种布局：

            * **HuggingFace 布局**（``fishaudio/fish-speech-1.5``）：
              ``config.json`` / ``model.pth`` /
              ``firefly-gan-vq-fsq-8x1024-21hz-generator.pth`` /
              ``tokenizer.tiktoken`` / ``special_tokens.json``。
            * **旧布局**：``vocab.json`` / ``acoustic_model.*`` /
              ``vq_decoder.safetensors`` / ``hifi_gan.safetensors``。

            若目录不存在或为空且 ``repo_id`` / ``backend_name`` 可用，
            将自动从 HuggingFace 下载模型。
        hifi_gan_path : str | None
            HiFi-GAN 权重路径；``None`` 时按候选列表查找
            （``firefly-gan-vq-fsq-8x1024-21hz-generator.pth`` 等）。
        audio_encoder_path : str | None
            AudioEncoder 权重路径（语音克隆用）；``None`` 时不加载
            AudioEncoder，语音克隆功能不可用。
        codec_type : str
            音频编码器类型，``"dac"`` / ``"encodec"`` / ``"snac"``。
        language : str
            默认语言代码。
        use_flash_attention : bool
            声学模型是否使用 Flash Attention。
        streaming_enabled : bool
            是否启用流式合成。
        scheduler : Any
            显存调度器实例。
        repo_id : str | None
            HuggingFace 仓库 ID（如 ``"fishaudio/fish-speech-1.5"``）。
            ``None`` 时使用 ``backend_name="fish"`` 对应的默认仓库。
        """
        super().__init__(scheduler=scheduler)

        self._model_path: str = model_path
        self._hifi_gan_path: str | None = hifi_gan_path
        self._audio_encoder_path: str | None = audio_encoder_path
        self._codec_type: str = codec_type
        self._language: str = language
        self._use_flash_attention: bool = use_flash_attention
        self._streaming_enabled: bool = streaming_enabled
        # HuggingFace 仓库 ID（用于自动下载模型）
        self._repo_id: str | None = repo_id

        # VQDecoder / HiFiGanVocoder / AudioEncoder 实例引用
        self._vq_decoder: Any = None
        self._hifi_gan: Any = None
        self._audio_encoder: Any = None

        # 从 config.json 读取的模型超参数（在 _build_pipeline 中填充）
        # Fish Speech HF 仓库 config.json 包含：model_type="dual_ar",
        # dim=1024, n_layer=24, codebook_size=1024, num_codebooks=8,
        # vocab_size=102048 等字段。
        self._hf_config: dict[str, Any] = {}
        self._config_codebook_size: int | None = None
        self._config_num_codebooks: int | None = None
        self._config_vocab_size: int | None = None
        self._config_dim: int | None = None
        self._config_n_layer: int | None = None

    # ==================================================================
    # 生命周期：组装 / 销毁管线
    # ==================================================================
    def _build_pipeline(self) -> None:
        """组装四层管线。

        依次构建并加载：

        1. Layer 1 — :class:`FishTokenizer`（文本前端）
        2. Layer 2 — :class:`FishLlamaARModel`（声学模型）
        3. Layer 3a — :class:`VQDecoder`（codec token → mel）
        4. Layer 3b — :class:`HiFiGanVocoder`（mel → waveform）
        5. Layer 3 — :class:`_CompositeVocoder`（组合 3a + 3b）
        6. （可选）AudioEncoder（语音克隆用）
        7. Layer 4 — :class:`StreamAdapter`（流式适配，按需构建）

        所有组件均在此方法内部延迟导入，避免模块加载阶段的硬依赖。
        """
        from mosaic.nodes.audio.tts_backends.acoustic_models.fish_ar import (
            FishLlamaARModel,
        )
        from mosaic.nodes.audio.tts_backends.hf_model_manager import HFModelManager
        from mosaic.nodes.audio.tts_backends.streaming.base import StreamAdapter
        from mosaic.nodes.audio.tts_backends.text_frontends.fish_tokenizer import (
            FishTokenizer,
        )
        from mosaic.nodes.audio.tts_backends.vocoders.hifi_gan import (
            HiFiGanVocoder,
        )
        from mosaic.nodes.audio.tts_backends.vocoders.vq_decoder import VQDecoder

        # ------------------------------------------------------------------
        # 确保 HF 模型已下载到本地
        # ------------------------------------------------------------------
        # Fish Speech HF 仓库 (fishaudio/fish-speech-1.5) 实际布局：
        #   config.json                                     模型配置
        #   model.pth                                       主 AR 模型权重
        #   firefly-gan-vq-fsq-8x1024-21hz-generator.pth    声码器 (FireflyGAN VQ generator)
        #   tokenizer.tiktoken                              TikToken 分词器
        #   special_tokens.json                             特殊 token 映射
        #
        # 若 model_path 已是存在的目录（含手动布置的模型目录），
        # 直接使用该目录；否则调用 ensure_model 自动从 HuggingFace 下载。
        if os.path.isdir(self._model_path):
            model_dir = self._model_path
        else:
            try:
                model_dir = HFModelManager.ensure_model(
                    self._model_path,
                    repo_id=self._repo_id,
                    backend_name="fish",
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "HF model download skipped/failed for %s: %s. "
                    "Using local path as-is.",
                    self._model_path,
                    exc,
                )
                model_dir = self._model_path

        # ------------------------------------------------------------------
        # 读取 config.json 获取模型超参数
        # ------------------------------------------------------------------
        config_path = os.path.join(model_dir, "config.json")
        hf_config = HFModelManager.load_json_config(config_path)
        self._hf_config = hf_config
        self._config_codebook_size = hf_config.get("codebook_size")
        self._config_num_codebooks = hf_config.get("num_codebooks")
        self._config_vocab_size = hf_config.get("vocab_size")
        self._config_dim = hf_config.get("dim")
        self._config_n_layer = hf_config.get("n_layer")
        if hf_config:
            self._logger.info(
                "Loaded Fish Speech config: codebook_size=%s, "
                "num_codebooks=%s, vocab_size=%s, dim=%s, n_layer=%s",
                self._config_codebook_size,
                self._config_num_codebooks,
                self._config_vocab_size,
                self._config_dim,
                self._config_n_layer,
            )

        # ------------------------------------------------------------------
        # 解析各组件权重路径（按 HF 仓库实际布局查找，兼容旧布局）
        # ------------------------------------------------------------------
        # 分词器：HF 仓库使用 tokenizer.tiktoken + special_tokens.json
        vocab_path = HFModelManager.find_file(
            model_dir,
            ["tokenizer.tiktoken", "tokenizer.json", "vocab.json"],
        )
        # 主模型权重：HF 仓库为单个 model.pth 文件
        main_model_path = HFModelManager.find_file(
            model_dir,
            ["model.pth", "model.safetensors", "pytorch_model.bin"],
        )
        # 声码器：HF 仓库为 firefly-gan-vq-fsq-8x1024-21hz-generator.pth
        if self._hifi_gan_path:
            vocoder_path = self._hifi_gan_path
        else:
            vocoder_path = HFModelManager.find_file(
                model_dir,
                [
                    "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                    "hifi_gan.safetensors",
                    "vocoder.pth",
                    "generator.pth",
                ],
            )

        # ------------------------------------------------------------------
        # Layer 1: 文本前端 —— FishTokenizer
        # ------------------------------------------------------------------
        # vocab_path 为空字符串时回退到字符级分词
        self._text_frontend = FishTokenizer(
            vocab_path=vocab_path,
            text_vocab_size=self._text_frontend_vocab_size(),
            audio_vocab_size=self._audio_vocab_size(),
            language=self._language,
        )

        # ------------------------------------------------------------------
        # Layer 2: 声学模型 —— FishLlamaARModel
        # ------------------------------------------------------------------
        # 主模型权重：优先使用 HF 单文件，回退到目录（兼容旧布局）
        main_weights = main_model_path or model_dir
        self._acoustic_model = FishLlamaARModel(
            model_path=model_dir,
            text_vocab_size=self._text_frontend.vocab_size
            - self._audio_vocab_size(),
            audio_vocab_size=self._audio_vocab_size(),
            use_flash_attention=self._use_flash_attention,
            codec_type=self._codec_type,
        )
        self._acoustic_model.load_weights(
            weights_path=main_weights,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # Layer 3a: VQ 解码器 —— codec token → mel
        # ------------------------------------------------------------------
        # VQ 解码器权重：优先查找单独文件；HF 仓库中 VQ 解码器与声码器合并在
        # firefly-gan-vq-fsq-8x1024-21hz-generator.pth 中，找不到单独文件时
        # 回退到主模型文件或声码器文件。
        self._vq_decoder = VQDecoder(
            codec_type=self._codec_type,
            codebook_size=self._audio_vocab_size(),
            codebook_dim=8,
            num_codebooks=self._num_codebooks(),
            hidden_size=512,
            mel_bins=80,
        )
        vq_path = HFModelManager.find_file(
            model_dir,
            ["vq_decoder.safetensors", "vq_decoder.pth", "vq_decoder.bin"],
        )
        if not vq_path:
            # 回退：从主模型文件或声码器文件中加载 VQ 解码器权重
            vq_path = main_model_path or vocoder_path or os.path.join(
                model_dir, "vq_decoder.safetensors"
            )
        self._vq_decoder.load_weights(
            weights_path=vq_path,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # Layer 3b: HiFi-GAN 声码器 —— mel → waveform
        # ------------------------------------------------------------------
        # 声码器权重：优先 HF firefly-gan 文件，回退到旧布局 hifi_gan.safetensors
        hifi_path = vocoder_path or os.path.join(
            model_dir, "hifi_gan.safetensors"
        )
        self._hifi_gan = HiFiGanVocoder(
            model_path=hifi_path,
            sample_rate=self.spec.sample_rate,
            n_mels=80,
        )
        self._hifi_gan.load_weights(
            weights_path=hifi_path,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # Layer 3: 复合声码器 —— 组合 VQDecoder + HiFi-GAN
        # ------------------------------------------------------------------
        self._vocoder = _CompositeVocoder(self._vq_decoder, self._hifi_gan)

        # ------------------------------------------------------------------
        # （可选）AudioEncoder —— 语音克隆用
        # ------------------------------------------------------------------
        if self._audio_encoder_path:
            self._load_audio_encoder()

        # ------------------------------------------------------------------
        # Layer 4: 流式适配器 —— 按需构建
        # ------------------------------------------------------------------
        if self._streaming_enabled:
            self._stream_adapter = StreamAdapter(
                chunk_size=4096,
                overlap=256,
                sample_rate=self.spec.sample_rate,
            )

    def _load_audio_encoder(self) -> None:
        """加载 AudioEncoder（语音克隆用）。

        AudioEncoder 用于将参考音频编码为 codec tokens。当前实现预留接口，
        实际加载逻辑取决于 codec_type（DAC / EnCodec / SNAC）。
        """
        self._logger.info(
            "AudioEncoder loading (codec_type=%s, path=%s) — interface "
            "placeholder, actual encoder loaded by FishLlamaARModel.",
            self._codec_type,
            self._audio_encoder_path,
        )
        # AudioEncoder 的实际实例由 FishLlamaARModel 内部管理
        # 此处仅记录路径，供 clone_voice 使用

    def _destroy_pipeline(self) -> None:
        """销毁四层管线并释放资源。"""
        super()._destroy_pipeline()

        for name, comp in (
            ("vq_decoder", self._vq_decoder),
            ("hifi_gan", self._hifi_gan),
        ):
            if comp is None:
                continue
            unload = getattr(comp, "unload_weights", None)
            if callable(unload):
                try:
                    unload()
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug(
                        "Error unloading %s weights: %s", name, exc
                    )
        self._vq_decoder = None
        self._hifi_gan = None
        self._audio_encoder = None

    # ==================================================================
    # 辅助：词表大小
    # ==================================================================
    def _text_frontend_vocab_size(self) -> int:
        """返回文本前端词表大小（文本 + 音频）。

        Fish Speech 的统一词表 = text_vocab + audio_vocab。
        优先使用 config.json 中的 ``vocab_size``，否则使用默认值。
        """
        if self._config_vocab_size is not None:
            return self._config_vocab_size
        return 12000  # 默认：10000 文本 + 2000 音频

    def _audio_vocab_size(self) -> int:
        """返回音频 codec 词表大小。

        优先使用 config.json 中的 ``codebook_size``，否则使用默认值。
        """
        if self._config_codebook_size is not None:
            return self._config_codebook_size
        return 2000  # 默认值

    def _num_codebooks(self) -> int:
        """返回 codec codebook 数量。

        优先使用 config.json 中的 ``num_codebooks``，否则使用默认值。
        """
        if self._config_num_codebooks is not None:
            return self._config_num_codebooks
        return 1  # 默认值

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
        3. 文本预处理（``FishTokenizer.preprocess``）。
        4. 说话人/参考音频编码（``FishTokenizer.encode_speaker``）。
        5. 分词（``FishTokenizer.tokenize``）→ token_ids。
        6. 合并推理参数。
        7. 声学模型生成（``FishLlamaARModel.generate``）→ audio_codec_ids。
        8. 复合声码器解码（``_CompositeVocoder.decode``）→ waveform。
        9. 构造 :class:`AudioData` 返回。

        Parameters
        ----------
        text : str
            待合成文本。
        speaker : str | None
            说话人标识。Fish Speech 中可以是参考音频文件路径或预编码的
            codec token ids；``None`` 使用默认音色。
        language : str
            语言代码（``"zh"`` / ``"en"`` / ``"ja"`` / ``"ko"``）。
        speed : float
            语速倍率。
        **kwargs : Any
            额外参数，透传给声学模型（``temperature`` / ``top_p`` /
            ``top_k`` / ``max_new_tokens`` 等）。

        Returns
        -------
        AudioData
            合成结果。

        Raises
        ------
        RuntimeError
            后端未加载。
        ValueError
            文本为空。
        """
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("synthesize requires a non-empty 'text' string.")

        self._logger.info(
            "synthesize: backend=%s language=%s speaker=%s speed=%.2f text_len=%d",
            self.name,
            language,
            speaker,
            speed,
            len(text),
        )

        # 3. 文本预处理
        processed_text = self._text_frontend.preprocess(text)

        # 4. 说话人/参考音频编码
        ref_audio_codes = self._text_frontend.encode_speaker(speaker)

        # 5. 分词（传入 ref_tokens 以构造语音克隆序列）
        token_ids = self._text_frontend.tokenize(
            processed_text,
            language=language,
            ref_tokens=ref_audio_codes,
        )

        # 6. 合并推理参数
        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        # 7. 声学模型生成
        audio_codec_ids = self._acoustic_model.generate(
            token_ids,
            ref_audio_codes,
            **params,
        )

        # 8. 复合声码器解码
        waveform, sample_rate = self._decode_full(audio_codec_ids)

        # 9. 构造 AudioData
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

        Parameters
        ----------
        text : str
            待合成文本。
        speaker : str | None
            说话人标识。
        language : str
            语言代码。
        speed : float
            语速倍率。
        chunk_size : int
            每个音频块的目标采样数。
        **kwargs : Any
            额外参数。

        Yields
        ------
        AudioData
            逐块音频数据。
        """
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "synthesize_stream requires a non-empty 'text' string."
            )

        if not self.spec.supports_streaming or self._stream_adapter is None:
            self._logger.info(
                "Backend %r streaming disabled; falling back to blocking.",
                self.name,
            )
            yield self.synthesize(
                text, speaker=speaker, language=language,
                speed=speed, **kwargs,
            )
            return

        self._logger.info(
            "synthesize_stream: backend=%s language=%s speaker=%s "
            "speed=%.2f chunk_size=%d text_len=%d",
            self.name, language, speaker, speed, chunk_size, len(text),
        )

        # 文本预处理与分词
        processed_text = self._text_frontend.preprocess(text)
        ref_audio_codes = self._text_frontend.encode_speaker(speaker)
        token_ids = self._text_frontend.tokenize(
            processed_text, language=language, ref_tokens=ref_audio_codes,
        )

        # 合并推理参数
        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)
        params["stream_batch"] = 24

        session = self._get_stream_session(chunk_size)

        for audio_codec_chunk in self._acoustic_model.generate_stream(
            token_ids, ref_audio_codes, **params,
        ):
            waveform_chunk, _sr = self._decode_chunk(audio_codec_chunk)
            self._stream_push(session, waveform_chunk)
            yield from self._stream_drain(
                session, text, speaker, language, speed
            )

        yield from self._stream_finish(
            session, text, speaker, language, speed
        )

    # ==================================================================
    # 语音克隆
    # ==================================================================
    def clone_voice(
        self,
        audio: Any,
        text: str,
        language: str = "zh",
        **kwargs: Any,
    ) -> AudioData:
        """语音克隆的便捷方法。

        输入参考音频 + 目标文本，合成与参考音频音色相同、内容为目标文本
        的语音。

        Parameters
        ----------
        audio : AudioData | str
            参考音频。可以是 :class:`AudioData` 实例或音频文件路径。
        text : str
            目标文本。
        language : str
            语言代码。
        **kwargs : Any
            额外参数。

        Returns
        -------
        AudioData
            克隆语音结果。
        """
        self._ensure_loaded()

        # 编码参考音频为 codec tokens
        if hasattr(audio, "waveform"):
            # AudioData 实例
            ref_tokens = self._acoustic_model.encode_reference_audio(audio)
        elif isinstance(audio, str):
            # 音频文件路径 — 作为 speaker 传入 synthesize
            return self.synthesize(
                text, speaker=audio, language=language, **kwargs
            )
        else:
            # 假设已经是 codec token ids
            ref_tokens = audio

        return self.synthesize(
            text, speaker=ref_tokens, language=language, **kwargs
        )

    # ==================================================================
    # 查询与说话人管理
    # ==================================================================
    def list_speakers(self) -> list[str]:
        """返回内置的说话人列表。

        Fish Speech 主要通过参考音频实现任意音色克隆，内置音色较少。

        Returns
        -------
        list[str]
            内置音色标识列表。
        """
        return list(self._BUILTIN_SPEAKERS)

    @classmethod
    def check_dependencies(cls) -> bool:
        """检查运行时依赖是否可用。

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
