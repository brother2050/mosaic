# mosaic/nodes/audio/tts_backends/implementations/sovits_backend.py
"""GPT-SoVITS 后端实现。

将 GPT-SoVITS 的三层组件组装为统一的 :class:`TTSBackend`，提供阻塞合成与
流式合成能力。GPT-SoVITS 采用「音素级 G2P → GPT-2 自回归声学模型 →
SoVITS 解码器（SemanticEncoder + Normalizing Flow + 条件 HiFi-GAN）」的管线，
输出 32kHz 单声道波形。

三层组装
--------
* Layer 1 — :class:`SoVITSTokenizer`：音素级文本前端（中文拼音 G2P、英文
  ARPAbet、语言标记插入、停顿标记）。
* Layer 2 — :class:`GPT2ARModel`：基于 GPT-2 的自回归声学模型，文本音素
  token → 语义 token（SSL 码本 index）。双路径 Embedding（text + semantic）。
* Layer 3 — :class:`SoVITSDecoder`：SoVITS 解码器（SemanticEncoder +
  PriorEncoder + Normalizing Flow + ConditionalHiFiGANDecoder），
  语义 token → 波形。
* Layer 4 — :class:`StreamAdapter`：流式缓冲与 chunk 切分。

与 ChatTTS / Fish 后端的差异
-----------------------------
* **采样率**：GPT-SoVITS 使用 32000Hz（ChatTTS 24000Hz，Fish 22050Hz）。
* **声学模型**：GPT-2 架构（ChatTTS / Fish 使用 LLaMA 架构）。
* **声码器**：SoVITS 解码器含 Normalizing Flow（ChatTTS 使用 DVAE+Vocos，
  Fish 使用 VQDecoder+HiFi-GAN）。
* **语音克隆**：通过参考音频的 SSL 语义 token 实现（ChatTTS 通过种子生成
  嵌入，Fish 通过 codec tokens 拼接）。
* **语言支持**：中、英、日、韩、粤语（支持粤语）。
* **许可证**：MIT（ChatTTS 为 CC BY-NC 4.0，Fish 为 Apache-2.0）。

目录结构约定
------------
支持两种布局，由 :meth:`_build_pipeline` 通过 :class:`HFModelManager`
自动识别（优先 HuggingFace 布局，找不到时回退到旧布局）。

**HuggingFace 布局**（``lj1995/GPT-SoVITS``）::

    model_path/
    ├── chinese-hubert-base/              # SSL 模型 (HuBERT)
    ├── chinese-roberta-wwm-ext-large/    # 文本编码器 (RoBERTa)
    ├── gsv-v2final-pretrained/           # v2 预训练 (.ckpt + .pth)
    ├── sv/                               # 说话人验证模型
    ├── s1bert25hz-2kh-*.ckpt             # GPT 模型 (根目录)
    ├── s2G488k.pth, s2D488k.pth          # SoVITS generator/decoder (根目录)
    └── hifigan_do_03357000               # HiFiGAN 声码器权重

**旧布局**（向后兼容）::

    model_path/
    ├── gpt/
    │   ├── config.json          # GPT-2 配置
    │   ├── model.safetensors    # GPT 权重
    │   └── vocab.json           # 音素词表
    ├── sovits/
    │   ├── config.json          # SoVITS 配置
    │   ├── model.safetensors    # SoVITS 权重
    │   └── ssl/                 # SSL 模型（可选）
    └── speaker/
        ├── embeddings.json      # 预计算的说话人嵌入
        └── reference/           # 参考音频目录

设计要点
--------
* ``torch`` / ``transformers`` / ``safetensors`` 等重依赖采用惰性导入。
* 三层组件均在 :meth:`_build_pipeline` 内部延迟导入。
* 参考音频的 SSL 编码是最耗时的预处理步骤，结果缓存在 ``_speaker_cache``。
* GPT 生成的语义 token 需经过 SoVITS 解码，两个阶段的延迟不同。
* 流式时建议 chunk 至少 16 个语义 token（SoVITS Flow 逆变换需要足够长度）。
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from mosaic.core.types import AudioData
from mosaic.nodes.audio._ref_audio_utils import load_reference_audio
from mosaic.nodes.audio.tts_backends.base import TTSBackend, TTSBackendSpec

__all__ = ["GPTSoVITSBackend"]


class GPTSoVITSBackend(TTSBackend):
    """GPT-SoVITS TTS 后端。

    将 GPT-SoVITS 的文本前端（:class:`SoVITSTokenizer`）、声学模型
    （:class:`GPT2ARModel`）、SoVITS 解码器（:class:`SoVITSDecoder`）与
    流式适配器（:class:`StreamAdapter`）组装为统一的 :class:`TTSBackend`，
    支持中英日韩粤语阻塞合成与流式合成，并通过参考音频实现零样本语音克隆。

    生命周期
    --------
    1. 构造后端实例（``is_loaded=False``）。
    2. 调用 :meth:`load` 加载三层管线（``is_loaded=True``）。
    3. 调用 :meth:`synthesize` / :meth:`synthesize_stream` 合成语音。
    4. 调用 :meth:`unload` 释放资源。

    Examples
    --------
    >>> backend = GPTSoVITSBackend(model_path="/data/gpt_sovits")
    >>> backend.load(device="cuda", dtype="float16")
    >>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

    Notes
    -----
    GPT-SoVITS 模型遵循 **MIT** 许可。输出采样率为 32000Hz。
    """

    name: str = "sovits"
    spec: TTSBackendSpec = TTSBackendSpec(
        name="sovits",
        supported_languages=["zh", "en", "ja", "ko", "yue"],
        supports_streaming=True,
        supports_voice_clone=True,
        vocoder_type="sovits_decoder",
        acoustic_type="ar",
        min_gpu_memory_gb=4.0,
        model_license="MIT",
        sample_rate=32000,
        default_params={
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.35,
            "max_new_tokens": 1024,
            "speed": 1.0,
        },
    )

    def __init__(
        self,
        model_path: str,
        gpt_path: str | None = None,
        sovits_path: str | None = None,
        ssl_model: str = "chinese-hubert-base",
        speaker_encoder_model: str = "default",
        language: str = "zh",
        streaming_enabled: bool = True,
        stream_batch: int = 16,
        scheduler: Any = None,
        repo_id: str | None = None,
    ) -> None:
        """初始化 GPT-SoVITS 后端。

        Parameters
        ----------
        model_path : str
            GPT-SoVITS 模型根路径。支持两种布局：

            * **HuggingFace 布局**（``lj1995/GPT-SoVITS``）：
              ``s1bert25hz-*.ckpt`` / ``s2G*.pth`` / ``s2D*.pth``
              / ``chinese-hubert-base/`` / ``chinese-roberta-wwm-ext-large/``
              / ``sv/`` / ``hifigan_*`` 等文件和目录。
            * **旧布局**：``gpt/`` 和 ``sovits/`` 子目录。

            若目录不存在或为空且 ``repo_id`` / ``backend_name`` 可用，
            将自动从 HuggingFace 下载模型。
        gpt_path : str | None
            GPT 模型单独路径；``None`` 时使用 ``model_path/gpt/``。
        sovits_path : str | None
            SoVITS 模型单独路径；``None`` 时使用 ``model_path/sovits/``。
        ssl_model : str
            SSL 模型名称或路径，默认 ``"chinese-hubert-base"``。
        speaker_encoder_model : str
            说话人编码器名称，默认 ``"default"``。
        language : str
            默认语言代码。
        streaming_enabled : bool
            是否启用流式合成。
        scheduler : Any
            显存调度器实例。
        repo_id : str | None
            HuggingFace 仓库 ID（如 ``"lj1995/GPT-SoVITS"``）。
            ``None`` 时使用 ``backend_name="sovits"`` 对应的默认仓库。
        """
        super().__init__(scheduler=scheduler)

        self._model_path: str = model_path
        self._gpt_path: str = gpt_path or os.path.join(model_path, "gpt")
        self._sovits_path: str = sovits_path or os.path.join(
            model_path, "sovits"
        )
        self._ssl_model_name: str = ssl_model
        self._speaker_encoder_model: str = speaker_encoder_model
        self._language: str = language
        self._streaming_enabled: bool = streaming_enabled
        # D2-2: 流式生成每次 yield 的语义 token 数（原硬编码 16）
        self._stream_batch: int = stream_batch
        # HuggingFace 仓库 ID（用于自动下载模型）
        self._repo_id: str | None = repo_id

        # SSL 模型实例（参考音频编码用）
        self._ssl_encoder: Any = None
        # 说话人特征缓存 {name: {"ref_semantic_tokens": ..., "speaker_embedding": ...}}
        self._speaker_cache: dict[str, dict[str, Any]] = {}
        # 预计算的说话人嵌入文件路径
        self._speaker_dir: str = os.path.join(model_path, "speaker")

        # HF 模型目录（_build_pipeline 中由 ensure_model 填充）
        self._model_dir: str = model_path
        # 各组件路径（_build_pipeline 中按 HF 布局查找填充）
        # SoVITS generator / decoder 权重文件路径
        self._sovits_g_path: str = ""
        self._sovits_d_path: str = ""
        # SSL 模型目录（chinese-hubert-base/）
        self._ssl_model_dir: str = ""
        # 文本编码器目录（chinese-roberta-wwm-ext-large/）
        self._text_encoder_dir: str = ""
        # 说话人验证模型文件路径（sv/pretrained_eres2netv2w24s4ep4.ckpt）
        self._speaker_verifier_path: str = ""
        # 声码器文件路径（hifigan_do_03357000 等）
        self._vocoder_path: str = ""

    # ==================================================================
    # 生命周期：组装 / 销毁管线
    # ==================================================================
    def _build_pipeline(self) -> None:
        """组装三层管线。

        依次构建并加载：

        1. Layer 1 — :class:`SoVITSTokenizer`（文本前端）
        2. Layer 2 — :class:`GPT2ARModel`（声学模型）
        3. Layer 3 — :class:`SoVITSDecoder`（SoVITS 解码器）
        4. （可选）SSL 模型（参考音频编码用）
        5. Layer 4 — :class:`StreamAdapter`（流式适配，按需构建）
        6. 加载预计算的说话人嵌入（如果有）

        所有组件均在此方法内部延迟导入，避免模块加载阶段的硬依赖。
        """
        from mosaic.nodes.audio.tts_backends.acoustic_models.gpt2_ar import (
            GPT2ARModel,
        )
        from mosaic.nodes.audio.tts_backends.hf_model_manager import HFModelManager
        from mosaic.nodes.audio.tts_backends.streaming.base import StreamAdapter
        from mosaic.nodes.audio.tts_backends.text_frontends.sovits_tokenizer import (
            SoVITSTokenizer,
        )
        from mosaic.nodes.audio.tts_backends.vocoders.sovits_decoder import (
            SoVITSDecoder,
        )

        # ------------------------------------------------------------------
        # 确保 HF 模型已下载到本地
        # ------------------------------------------------------------------
        # GPT-SoVITS HF 仓库 (lj1995/GPT-SoVITS) 实际布局：
        #   chinese-hubert-base/              SSL 模型 (HuBERT)
        #   chinese-roberta-wwm-ext-large/    文本编码器 (RoBERTa)
        #   gsv-v2final-pretrained/           v2 final 预训练 (GPT .ckpt + SoVITS .pth)
        #   gsv-v4-pretrained/                v4 预训练
        #   v2Pro/                            v2 Pro
        #   sv/                               说话人验证模型
        #   s1bert25hz-2kh-*.ckpt             GPT 模型 v2 (根目录)
        #   s1v3.ckpt                         GPT 模型 v3 (根目录)
        #   s2G488k.pth, s2D488k.pth          SoVITS v1 (根目录)
        #   s2Gv3.pth                         SoVITS v3 (根目录)
        #   hifigan_config.json, hifigan_do_03357000   HiFiGAN 配置与权重
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
                    backend_name="sovits",
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "HF model download skipped/failed for %s: %s. "
                    "Using local path as-is.",
                    self._model_path,
                    exc,
                )
                model_dir = self._model_path
        self._model_dir = model_dir

        # ------------------------------------------------------------------
        # 解析各组件路径（按 HF 仓库实际布局查找，兼容旧 gpt/ sovits/ 布局）
        # ------------------------------------------------------------------
        # GPT 模型权重：HF 仓库为 .ckpt 文件（如 s1bert25hz-2kh-*.ckpt）
        gpt_ckpt_path = HFModelManager.find_file(
            model_dir,
            [
                "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                "s1v3.ckpt",
                "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            ],
        )
        # 回退：未找到 .ckpt 时使用旧 gpt/ 目录（保持兼容）
        gpt_weights = gpt_ckpt_path or self._gpt_path

        # SoVITS 解码器权重：HF 仓库为 .pth 文件
        # s2G* 为 generator，s2D* 为 decoder
        sovits_g_path = HFModelManager.find_file(
            model_dir,
            [
                "s2G488k.pth",
                "s2Gv3.pth",
                "gsv-v2final-pretrained/s2G2333k.pth",
                "v2Pro/s2Gv2Pro.pth",
            ],
        )
        sovits_d_path = HFModelManager.find_file(
            model_dir,
            [
                "s2D488k.pth",
                "gsv-v2final-pretrained/s2D2333k.pth",
                "v2Pro/s2Dv2Pro.pth",
            ],
        )
        self._sovits_g_path = sovits_g_path
        self._sovits_d_path = sovits_d_path
        # 回退：未找到 .pth 时使用旧 sovits/ 目录（保持兼容）
        sovits_weights = sovits_g_path or self._sovits_path

        # SSL 模型目录：HF 仓库在 chinese-hubert-base/
        self._ssl_model_dir = HFModelManager.find_dir(
            model_dir,
            ["chinese-hubert-base", "sovits/ssl", "ssl"],
        )

        # 文本编码器目录：HF 仓库在 chinese-roberta-wwm-ext-large/
        self._text_encoder_dir = HFModelManager.find_dir(
            model_dir,
            ["chinese-roberta-wwm-ext-large", "bert", "text_encoder"],
        )

        # 说话人验证模型：HF 仓库在 sv/
        self._speaker_verifier_path = HFModelManager.find_file(
            model_dir,
            [
                "sv/pretrained_eres2netv2w24s4ep4.ckpt",
                "speaker_encoder/pretrained_eres2netv2w24s4ep4.ckpt",
            ],
        )

        # 声码器：HF 仓库为 hifigan_do_03357000 或 gsv-v4-pretrained/vocoder.pth
        self._vocoder_path = HFModelManager.find_file(
            model_dir,
            ["hifigan_do_03357000", "gsv-v4-pretrained/vocoder.pth", "vocoder.pth"],
        )

        # ------------------------------------------------------------------
        # Layer 1: 文本前端 —— SoVITSTokenizer
        # ------------------------------------------------------------------
        # 分词器：优先旧 gpt/ 目录，其次按 HF 仓库布局查找
        vocab_path = os.path.join(self._gpt_path, "vocab.json")
        if not os.path.isfile(vocab_path):
            vocab_path = HFModelManager.find_file(
                model_dir,
                [
                    "chinese-roberta-wwm-ext-large/tokenizer.json",
                    "gpt/vocab.json",
                    "vocab.json",
                ],
            )
        self._text_frontend = SoVITSTokenizer(
            vocab_path=vocab_path,
            language=self._language,
        )

        # ------------------------------------------------------------------
        # Layer 2: 声学模型 —— GPT2ARModel
        # ------------------------------------------------------------------
        # GPT 权重：优先 HF .ckpt 文件，回退到旧 gpt/ 目录
        self._acoustic_model = GPT2ARModel(
            model_path=self._gpt_path,
            vocab_size=self._text_frontend.vocab_size,
            semantic_vocab_size=768,  # SSL 码本大小
        )
        self._acoustic_model.load_weights(
            weights_path=gpt_weights,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # Layer 3: SoVITS 解码器 —— 语义 token → 波形
        # ------------------------------------------------------------------
        # SoVITS 权重：优先 HF .pth 文件（generator），回退到旧 sovits/ 目录
        self._vocoder = SoVITSDecoder(
            model_path=self._sovits_path,
            ssl_vocab_size=768,
            hidden_size=192,
            sample_rate=self.spec.sample_rate,
        )
        self._vocoder.load_weights(
            weights_path=sovits_weights,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # （可选）SSL 模型 —— 参考音频编码
        # ------------------------------------------------------------------
        self._load_ssl_model()

        # ------------------------------------------------------------------
        # Layer 4: 流式适配器 —— 按需构建
        # ------------------------------------------------------------------
        if self._streaming_enabled:
            self._stream_adapter = StreamAdapter(
                chunk_size=4096,
                overlap=256,
                sample_rate=self.spec.sample_rate,
            )

        # ------------------------------------------------------------------
        # 加载预计算的说话人嵌入
        # ------------------------------------------------------------------
        self._load_speaker_cache()

    def _destroy_pipeline(self) -> None:
        """销毁三层管线并释放资源。"""
        super()._destroy_pipeline()

        # 释放 SSL 模型
        if self._ssl_encoder is not None:
            unload = getattr(self._ssl_encoder, "unload_weights", None)
            if callable(unload):
                try:
                    unload()
                except Exception as exc:  # noqa: BLE001
                    self._logger.debug(
                        "Error unloading SSL encoder: %s", exc
                    )
        self._ssl_encoder = None
        self._speaker_cache.clear()

    def _load_ssl_model(self) -> None:
        """加载 SSL 模型用于参考音频编码。

        尝试通过 transformers 加载 HuBERT/Wav2Vec2 模型。如果加载失败，
        语音克隆功能将不可用（但普通合成不受影响）。

        SSL 模型路径解析优先级：

        1. HF 仓库布局：``chinese-hubert-base/`` 目录（由
           :meth:`_build_pipeline` 中 ``HFModelManager.find_dir`` 查找）。
        2. 旧布局：``sovits/ssl/`` 目录。
        3. ``self._ssl_model_name`` 模型名称（由 transformers 从网络拉取）。
        """
        # 优先使用 HF 仓库布局查找到的 SSL 目录
        if self._ssl_model_dir:
            ssl_path = self._ssl_model_dir
        else:
            # 回退到旧布局：sovits/ssl 目录或模型名称
            ssl_dir = os.path.join(self._sovits_path, "ssl")
            ssl_path = (
                ssl_dir if os.path.isdir(ssl_dir) else self._ssl_model_name
            )

        try:
            import torch  # type: ignore  # noqa: F401
            from transformers import (  # type: ignore
                AutoModel,
                AutoFeatureExtractor,
            )

            self._ssl_encoder = {
                "model": AutoModel.from_pretrained(ssl_path),
                "extractor": AutoFeatureExtractor.from_pretrained(ssl_path),
            }
            self._logger.info(
                "SSL model loaded from %s for reference audio encoding.",
                ssl_path,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load SSL model from %s: %s. "
                "Voice clone will be unavailable.",
                ssl_path,
                exc,
            )
            self._ssl_encoder = None

    def _load_speaker_cache(self) -> None:
        """加载预计算的说话人嵌入。"""
        embeddings_path = os.path.join(self._speaker_dir, "embeddings.json")
        if not os.path.isfile(embeddings_path):
            return
        try:
            with open(embeddings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._speaker_cache = data
                self._logger.info(
                    "Loaded %d pre-computed speaker embeddings.",
                    len(self._speaker_cache),
                )
        except Exception as exc:  # noqa: BLE001
            # E3-1: 说话人缓存加载失败应可见，而非静默 debug
            self._logger.warning(
                "Failed to load speaker cache: %s", exc, exc_info=True
            )

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
        3. 参考音频处理（如果有 speaker）：
           a. 加载参考音频
           b. SSL_Encoder(ref_audio) → ref_semantic_tokens
           c. SpeakerEncoder(ref_audio) → speaker_embedding
        4. SoVITSTokenizer.preprocess(text)
        5. phoneme_ids = SoVITSTokenizer.tokenize(text, language)
        6. 构造 speaker_info = {"ref_semantic_tokens": ..., "speaker_embedding": ...}
        7. semantic_tokens = GPT2ARModel.generate(phoneme_ids, speaker_info, ...)
        8. SoVITSDecoder.set_reference(ref_features, speaker_embedding)
        9. waveform = SoVITSDecoder.decode(semantic_tokens)
        10. 如果 speed != 1.0，做时间拉伸处理
        11. 返回 AudioData(waveform, sample_rate=32000)

        Parameters
        ----------
        text : str
            待合成文本。
        speaker : str | None
            说话人名称或参考音频路径；``None`` 使用默认音色。
        language : str
            语言代码（``"zh"`` / ``"en"`` / ``"ja"`` / ``"ko"`` / ``"yue"``）。
        speed : float
            语速倍率，``1.0`` 为正常语速。
        **kwargs : Any
            额外参数，透传给声学模型（``temperature`` / ``top_p`` /
            ``top_k`` / ``max_new_tokens`` 等）。

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
        self._ensure_loaded()
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

        # 3. 参考音频处理
        speaker_info = self._get_speaker_info(speaker)

        # 4-5. 文本前端处理
        processed = self._text_frontend.preprocess(text)
        token_ids = self._text_frontend.tokenize(processed, language=language)

        # 6. 合并推理参数
        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        # 7. 声学模型生成 —— semantic_tokens
        semantic_tokens = self._acoustic_model.generate(
            token_ids,
            speaker_embedding=speaker_info,
            **params,
        )

        # 8. 设置参考特征到 SoVITS 解码器
        if speaker_info and speaker_info.get("ref_semantic_tokens") is not None:
            self._vocoder.set_reference(
                speaker_info["ref_semantic_tokens"],
                speaker_info.get("speaker_embedding"),
            )

        # 9. 解码 —— semantic_tokens → waveform
        waveform, sample_rate = self._decode_full(semantic_tokens)

        # 10. 语速调整
        if speed != 1.0 and speed > 0:
            waveform = self._adjust_speed(waveform, speed)

        # 11. 构造 AudioData
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

        1. 参考音频处理同 :meth:`synthesize`。
        2. 文本前端处理同 :meth:`synthesize`。
        3. 创建 StreamSession。
        4. GPT2ARModel.generate_stream(phoneme_ids, speaker_info, stream_batch=16)
           - 每 16 个语义 token yield 一次。
        5. SoVITSDecoder.decode_chunk(semantic_chunk) → waveform chunk。
        6. StreamSession.push(waveform_chunk)。
        7. yield AudioData chunks。

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
            逐块音频数据，``metadata`` 中 ``streaming=True``。
        """
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "synthesize_stream requires a non-empty 'text' string."
            )

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

        # 参考音频处理
        speaker_info = self._get_speaker_info(speaker)

        # 文本前端处理
        processed = self._text_frontend.preprocess(text)
        token_ids = self._text_frontend.tokenize(processed, language=language)

        # 合并推理参数
        params: dict[str, Any] = dict(self.spec.default_params)
        params["stream_batch"] = self._stream_batch  # D2-2: 可配置（默认 16，GPT-SoVITS 建议至少 16 个语义 token；kwargs 可覆盖）
        params.update(kwargs)

        # 设置参考特征
        if speaker_info and speaker_info.get("ref_semantic_tokens") is not None:
            self._vocoder.set_reference(
                speaker_info["ref_semantic_tokens"],
                speaker_info.get("speaker_embedding"),
            )

        # 重置流式解码器状态
        self._vocoder.reset_stream()

        # 创建流式会话
        session = self._get_stream_session(chunk_size)

        try:
            # 流式生成 → 逐块解码 → 缓冲输出
            for semantic_chunk in self._acoustic_model.generate_stream(
                token_ids,
                speaker_embedding=speaker_info,
                **params,
            ):
                # 流式取消：提前终止生成循环
                if session.is_cancelled is True:
                    self._logger.info(
                        "synthesize_stream cancelled for backend %s",
                        self.name,
                    )
                    break
                waveform_chunk, _sr = self._decode_chunk(semantic_chunk)
                self._stream_push(session, waveform_chunk)
                yield from self._stream_drain(
                    session, text, speaker, language, speed
                )

            yield from self._stream_finish(
                session, text, speaker, language, speed
            )
        finally:
            # 确保会话缓冲与声学模型 KV cache 被释放/重置（即使中途抛异常）
            self._cleanup_stream_state(session)

    # ==================================================================
    # 语音克隆
    # ==================================================================
    def clone_voice(
        self,
        audio: AudioData | str,
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

        # 提取参考音频特征
        speaker_info = self.extract_speaker(audio)

        # 使用提取的特征进行合成
        return self._synthesize_with_speaker_info(
            text, speaker_info, language, **kwargs
        )

    def clone_voice_stream(
        self,
        audio: AudioData | str,
        text: str,
        language: str = "zh",
        **kwargs: Any,
    ) -> Iterator[AudioData]:
        """流式语音克隆。

        Parameters
        ----------
        audio : AudioData | str
            参考音频。
        text : str
            目标文本。
        language : str
            语言代码。
        **kwargs : Any
            额外参数。

        Yields
        ------
        AudioData
            逐块音频数据。
        """
        self._ensure_loaded()

        # 提取参考音频特征
        speaker_info = self.extract_speaker(audio)

        # 流式合成
        yield from self._synthesize_stream_with_speaker_info(
            text, speaker_info, language, **kwargs
        )

    # ==================================================================
    # 说话人管理
    # ==================================================================
    def list_speakers(self) -> list[str]:
        """返回预计算的说话人列表。

        GPT-SoVITS 的优势是极少样本克隆，内置音色可能不多。

        Returns
        -------
        list[str]
            已保存的说话人名称列表。
        """
        return list(self._speaker_cache.keys())

    def extract_speaker(
        self, audio: AudioData | str
    ) -> dict[str, Any]:
        """从音频中提取说话人特征。

        提取 ``ref_semantic_tokens``（SSL 编码的语义 token）和
        ``speaker_embedding``（说话人嵌入向量）。可以预计算并缓存，
        避免每次合成重复计算。

        Parameters
        ----------
        audio : AudioData | str
            参考音频。可以是 :class:`AudioData` 实例或音频文件路径。

        Returns
        -------
        dict[str, Any]
            包含 ``ref_semantic_tokens`` 和 ``speaker_embedding`` 的字典。

        Raises
        ------
        RuntimeError
            SSL 模型未加载。
        """
        self._ensure_loaded()

        # 获取波形
        waveform = self._get_waveform(audio)
        if waveform is None:
            return {"ref_semantic_tokens": None, "speaker_embedding": None}

        # SSL 编码 → 语义 token
        ref_semantic_tokens = self._encode_with_ssl(waveform)

        # 说话人嵌入（简化：使用语义 token 的统计量作为嵌入）
        speaker_embedding = self._compute_speaker_embedding(ref_semantic_tokens)

        return {
            "ref_semantic_tokens": ref_semantic_tokens,
            "speaker_embedding": speaker_embedding,
        }

    def save_speaker(
        self, name: str, audio: AudioData | str
    ) -> None:
        """提取并保存说话人特征到本地。

        Parameters
        ----------
        name : str
            说话人名称。
        audio : AudioData | str
            参考音频。
        """
        self._ensure_loaded()

        speaker_info = self.extract_speaker(audio)
        self._speaker_cache[name] = speaker_info

        # 持久化到文件
        os.makedirs(self._speaker_dir, exist_ok=True)
        embeddings_path = os.path.join(self._speaker_dir, "embeddings.json")

        # 将 tensor 转为 list 以便 JSON 序列化
        serializable: dict[str, Any] = {}
        for spk_name, info in self._speaker_cache.items():
            entry: dict[str, Any] = {}
            for k, v in info.items():
                if hasattr(v, "tolist"):
                    entry[k] = v.tolist()
                elif isinstance(v, list):
                    entry[k] = v
                else:
                    entry[k] = v
            serializable[spk_name] = entry

        try:
            with open(embeddings_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            self._logger.info(
                "Saved speaker %r to %s.", name, embeddings_path
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to save speaker: %s", exc)

    def load_speaker(self, name: str) -> dict[str, Any]:
        """加载已保存的说话人特征。

        Parameters
        ----------
        name : str
            说话人名称。

        Returns
        -------
        dict[str, Any]
            包含 ``ref_semantic_tokens`` 和 ``speaker_embedding`` 的字典。

        Raises
        ------
        KeyError
            说话人名称不存在。
        """
        if name not in self._speaker_cache:
            raise KeyError(f"Speaker {name!r} not found in cache.")
        return self._speaker_cache[name]

    # ==================================================================
    # 依赖检查
    # ==================================================================
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

    # ==================================================================
    # 内部辅助
    # ==================================================================
    def _get_speaker_info(
        self, speaker: str | None
    ) -> dict[str, Any] | None:
        """获取说话人信息。

        优先从缓存查找；如果是音频路径则实时提取。

        Parameters
        ----------
        speaker : str | None
            说话人名称或参考音频路径。

        Returns
        -------
        dict[str, Any] | None
            包含 ``ref_semantic_tokens`` 和 ``speaker_embedding`` 的字典。
        """
        if speaker is None:
            return None

        # 1. 从缓存查找
        if speaker in self._speaker_cache:
            return self._speaker_cache[speaker]

        # 2. 如果是文件路径，实时提取
        if isinstance(speaker, str) and os.path.isfile(speaker):
            try:
                return self.extract_speaker(speaker)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to extract speaker from %s: %s",
                    speaker,
                    exc,
                    exc_info=True,
                )
                return None

        # 3. 未知 speaker：A3-4 记录警告，回退默认音色（避免静默）
        self._logger.warning(
            "Speaker %r not found in cache or as a file path; "
            "falling back to default speaker.",
            speaker,
        )
        return None

    def _get_waveform(self, audio: AudioData | str) -> Any:
        """从 AudioData 或文件路径获取波形。

        使用统一的参考音频预处理工具 :func:`load_reference_audio` 加载、
        校验时长并自动截断到 GPT-SoVITS 推荐时长上限（10 秒），避免超长
        参考音频导致 OOM 或晦涩错误。GPT-SoVITS 模型采样率为 32000Hz。
        """
        is_audio_data = hasattr(audio, "waveform") and hasattr(
            audio, "sample_rate"
        )
        is_file = isinstance(audio, str) and os.path.isfile(audio)
        if not (is_audio_data or is_file):
            return None
        try:
            waveform, _sr = load_reference_audio(
                audio, target_sr=32000, backend="sovits"
            )
            return waveform
        except FileNotFoundError as exc:
            self._logger.warning("Reference audio file not found: %s", exc)
            return None
        except ValueError as exc:
            self._logger.warning("Invalid reference audio: %s", exc)
            return None
        except ImportError as exc:
            self._logger.warning(
                "soundfile/librosa not installed; cannot load audio file: %s",
                exc,
            )
            return None

    def _encode_with_ssl(self, waveform: Any) -> Any:
        """使用 SSL 模型编码波形为语义 token。

        Parameters
        ----------
        waveform : Any
            音频波形（由 :meth:`_get_waveform` 加载，采样率为 32000Hz）。

        Returns
        -------
        Any
            语义 token ids，或 ``None``（SSL 模型未加载时）。

        Notes
        -----
        ``chinese-hubert-base`` 以 **16kHz** 训练，而参考音频被加载为
        32000Hz。送入 HuBERT 前必须重采样到 16kHz，否则采样率不匹配会
        导致特征帧率错位、克隆音色失真。
        """
        if self._ssl_encoder is None:
            return None

        try:
            import torch  # type: ignore

            model = self._ssl_encoder["model"]
            extractor = self._ssl_encoder["extractor"]

            # 确保模型在正确设备上
            device = self._device
            model = model.to(device)
            model.eval()

            # C2-1: chinese-hubert-base 以 16kHz 训练，重采样到 16kHz
            # 后再送入 HuBERT，避免采样率不匹配导致特征错位。
            waveform_16k = self._resample_to_16k(waveform, orig_sr=32000)

            # 提取特征（告知 extractor 实际为 16kHz）
            inputs = extractor(
                waveform_16k, sampling_rate=16000, return_tensors="pt"
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values)
                # 取最后一层隐藏状态
                hidden_states = outputs.last_hidden_state  # [1, T, H]

            # 通过 k-means 或简单的 argmax 量化为语义 token
            # E4-3 NOTE: 当前使用 argmax 量化，与原始 GPT-SoVITS 的 k-means
            # 语义 token 不同。这可能影响 zero-shot 语音克隆的质量。
            # 完整 k-means 量化需要预训练码本，此处仅作最小修复（标注限制）。
            self._logger.debug(
                "Using argmax quantization for SSL features (not k-means). "
                "Voice cloning quality may be affected."
            )
            semantic_tokens = hidden_states.argmax(dim=-1)  # [1, T]
            return semantic_tokens
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "SSL encoding failed: %s", exc, exc_info=True
            )
            return None

    @staticmethod
    def _resample_to_16k(waveform: Any, orig_sr: int) -> Any:
        """将波形重采样到 16kHz（供 HuBERT 使用）。

        优先使用 ``librosa.resample``；未安装 librosa 时回退到线性插值。
        输入为 numpy 数组或 torch 张量。
        """
        if orig_sr == 16000:
            return waveform
        try:
            import numpy as np

            # 统一转为 1D numpy 处理
            is_tensor = False
            import torch  # type: ignore

            if isinstance(waveform, torch.Tensor):
                is_tensor = True
                arr = waveform.detach().cpu().numpy()
            else:
                arr = np.asarray(waveform, dtype=np.float32)
            arr = np.atleast_1d(arr).ravel()

            try:
                import librosa

                resampled = librosa.resample(arr, orig_sr=orig_sr, target_sr=16000)
            except ImportError:
                # 线性插值回退
                ratio = 16000 / orig_sr
                n_samples = max(1, int(len(arr) * ratio))
                indices = np.linspace(0, len(arr) - 1, n_samples)
                resampled = np.interp(
                    indices, np.arange(len(arr)), arr
                ).astype(np.float32)

            if is_tensor:
                return torch.from_numpy(np.asarray(resampled))
            return resampled
        except ImportError:
            # torch/numpy 均不可用，原样返回（下游会报错）
            return waveform

    def _compute_speaker_embedding(self, semantic_tokens: Any) -> Any:
        """从语义 token 计算说话人嵌入。

        简化实现：使用语义 token 的统计量（均值 + 标准差）作为嵌入。

        .. warning::

           E4-3: 当前使用 ``one-hot + 均值池化`` 作为说话人嵌入，与原始
           GPT-SoVITS 的说话人验证模型（ERES2Net）嵌入语义不同，几乎无
           说话人区分力，可能影响克隆音色相似度。完整实现应使用
           ``sv/pretrained_eres2netv2w24s4ep4.ckpt`` 说话人验证模型。

        Parameters
        ----------
        semantic_tokens : Any
            语义 token ids。

        Returns
        -------
        Any
            说话人嵌入向量。
        """
        if semantic_tokens is None:
            return None

        try:
            import torch  # type: ignore

            if isinstance(semantic_tokens, torch.Tensor):
                self._logger.debug(
                    "Using one-hot mean as speaker embedding (not a real "
                    "speaker verifier model); speaker discrimination is weak."
                )
                # 使用 one-hot + 均值池化
                tokens = semantic_tokens.long()
                one_hot = torch.nn.functional.one_hot(
                    tokens, num_classes=768
                ).float()
                embedding = one_hot.mean(dim=1)  # [1, 768]
                return embedding
        except Exception as exc:  # noqa: BLE001
            # E3-1: 不再静默吞掉异常
            self._logger.warning(
                "Failed to compute speaker embedding: %s", exc, exc_info=True
            )

        return None

    def _synthesize_with_speaker_info(
        self,
        text: str,
        speaker_info: dict[str, Any],
        language: str = "zh",
        **kwargs: Any,
    ) -> AudioData:
        """使用指定的说话人信息进行合成。"""
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("synthesize requires a non-empty 'text' string.")

        processed = self._text_frontend.preprocess(text)
        token_ids = self._text_frontend.tokenize(processed, language=language)

        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        semantic_tokens = self._acoustic_model.generate(
            token_ids,
            speaker_embedding=speaker_info,
            **params,
        )

        if speaker_info.get("ref_semantic_tokens") is not None:
            self._vocoder.set_reference(
                speaker_info["ref_semantic_tokens"],
                speaker_info.get("speaker_embedding"),
            )

        waveform, sample_rate = self._decode_full(semantic_tokens)

        duration = self._compute_duration(waveform, sample_rate)
        return AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            metadata={
                "backend": self.name,
                "text": text,
                "speaker": "cloned",
                "language": language,
                "speed": params.get("speed", 1.0),
                "duration": duration,
                "sample_rate": sample_rate,
                "streaming": False,
            },
        )

    def _synthesize_stream_with_speaker_info(
        self,
        text: str,
        speaker_info: dict[str, Any],
        language: str = "zh",
        **kwargs: Any,
    ) -> Iterator[AudioData]:
        """使用指定的说话人信息进行流式合成。"""
        self._ensure_loaded()
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "synthesize_stream requires a non-empty 'text' string."
            )

        processed = self._text_frontend.preprocess(text)
        token_ids = self._text_frontend.tokenize(processed, language=language)

        params: dict[str, Any] = dict(self.spec.default_params)
        params["stream_batch"] = self._stream_batch  # D2-2: 可配置（kwargs 可覆盖）
        params.update(kwargs)

        if speaker_info.get("ref_semantic_tokens") is not None:
            self._vocoder.set_reference(
                speaker_info["ref_semantic_tokens"],
                speaker_info.get("speaker_embedding"),
            )
        self._vocoder.reset_stream()

        session = self._get_stream_session(kwargs.get("chunk_size", 4096))

        try:
            for semantic_chunk in self._acoustic_model.generate_stream(
                token_ids,
                speaker_embedding=speaker_info,
                **params,
            ):
                waveform_chunk, _sr = self._decode_chunk(semantic_chunk)
                self._stream_push(session, waveform_chunk)
                yield from self._stream_drain(
                    session, text, "cloned", language, params.get("speed", 1.0)
                )

            yield from self._stream_finish(
                session, text, "cloned", language, params.get("speed", 1.0)
            )
        finally:
            # 确保会话缓冲与声学模型 KV cache 被释放/重置（即使中途抛异常）
            self._cleanup_stream_state(session)

    @staticmethod
    def _adjust_speed(waveform: Any, speed: float) -> Any:
        """调整语速（时间拉伸）。

        使用简单的重采样方法：当 speed > 1.0 时加速（下采样），
        speed < 1.0 时减速（上采样）。

        Parameters
        ----------
        waveform : Any
            输入波形。
        speed : float
            语速倍率。

        Returns
        -------
        Any
            调整后的波形。
        """
        try:
            import torch  # type: ignore

            if isinstance(waveform, torch.Tensor):
                if waveform.dim() == 1:
                    # 一维波形
                    n = waveform.shape[0]
                    new_n = max(1, int(n / speed))
                    indices = torch.linspace(
                        0, n - 1, new_n, device=waveform.device
                    ).long()
                    return waveform[indices]
                elif waveform.dim() == 2:
                    # 二维波形 [B, T]
                    n = waveform.shape[-1]
                    new_n = max(1, int(n / speed))
                    indices = torch.linspace(
                        0, n - 1, new_n, device=waveform.device
                    ).long()
                    return waveform[..., indices]
        except ImportError:
            pass

        # numpy 回退
        try:
            import numpy as np  # type: ignore

            if isinstance(waveform, np.ndarray):
                n = waveform.shape[-1]
                new_n = max(1, int(n / speed))
                indices = np.linspace(0, n - 1, new_n).astype(int)
                return waveform[..., indices]
        except ImportError:
            pass

        # 无法处理，原样返回
        return waveform
