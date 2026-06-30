# mosaic/nodes/audio/tts_backends/implementations/cosyvoice_backend.py
"""CosyVoice 后端实现。

将 CosyVoice 的组件组装为统一的 :class:`TTSBackend`，提供阻塞合成与
分块流式合成能力。CosyVoice 采用「LLM 文本理解 → Flow Matching ODE 求解 →
HiFi-GAN 声码器」的管线，输出 24000Hz 单声道波形。

与前三者的根本差异
------------------
* **声学模型**：Flow Matching（非自回归），通过 ODE 求解从高斯噪声一次性
  生成完整 mel spectrogram（ChatTTS / Fish / GPT-SoVITS 使用自回归）。
* **生成方式**：非自回归，不逐 token 产出；流式通过 Chunk-aware ODE 实现。
* **文本前端**：使用预训练 LLM 的 tokenizer（Qwen2.5），而非自定义 BPE
  或音素 G2P。
* **采样率**：24000Hz（与 cosyvoice2.yaml 一致；ChatTTS 24000Hz，GPT-SoVITS 32000Hz）。
* **许可证**：Apache-2.0（ChatTTS 为 CC BY-NC 4.0）。

管线
----
::

    Phase 1: 文本理解
      text → CosyVoiceTokenizer → token_ids → LLM → text_hidden_states
      → text_projection → text_feats [batch, feat_dim, text_len]

    Phase 2: 参考音频编码
      ref_audio → SpeechTokenizer → ref_speech_tokens [1, ref_len]
      ref_audio → SpeakerEncoder → speaker_embedding [1, spk_dim]

    Phase 3: 条件融合
      condition = fuse(text_feats, ref_speech_feats, speaker_embedding)

    Phase 4: Flow Matching ODE 求解
      z_T ~ N(0, I) → ODE 求解 → mel

    Phase 5: HiFi-GAN 声码器
      mel → HiFiGanVocoder → waveform

HuggingFace 仓库布局（FunAudioLLM/CosyVoice2-0.5B）
--------------------------------------------------
::

    model_path/
    ├── CosyVoice-BlankEN/        # LLM tokenizer 模型（Qwen2, 896 dim）
    │   ├── config.json
    │   ├── merges.txt / vocab.json
    │   ├── tokenizer_config.json
    │   └── model.safetensors
    ├── campplus.onnx              # 说话人编码器（CAM++，spk_embed_dim=192）
    ├── cosyvoice2.yaml            # 主配置（sample_rate=24000, spk_embed_dim=192）
    ├── flow.decoder.estimator.fp32.onnx
    ├── flow.pt                    # Flow Matching 权重
    ├── hift.pt                    # HiFi-GAN 声码器权重（HiFTGenerator）
    ├── llm.pt                     # LLM 权重（Qwen2）
    ├── speech_tokenizer_v2.onnx   # 语音 tokenizer
    ├── speech_tokenizer_v2.batch.onnx
    └── speaker/                   # 预计算说话人嵌入（可选，向后兼容）
        ├── embeddings.json
        └── reference/

旧版目录布局（``model_path/llm``、``model_path/flow_matching`` 等）仍被
向后兼容地支持：当 HF 布局的文件未找到时，回退到旧目录。

设计要点
--------
* ``torch`` / ``transformers`` / ``safetensors`` 等重依赖采用惰性导入。
* 参考音频的 SSL 编码是最耗时的预处理步骤，结果缓存在 ``_speaker_cache``。
* ODE 步数可运行时调整（:meth:`set_ode_params`），支持质量/速度权衡。
* HiFiGanVocoder 与 Fish Speech 共享代码，权重独立。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

from mosaic.core.types import AudioData
from mosaic.nodes.audio._ref_audio_utils import load_reference_audio
from mosaic.nodes.audio.tts_backends.base import TTSBackend, TTSBackendSpec

__all__ = ["CosyVoiceBackend"]


class CosyVoiceBackend(TTSBackend):
    """CosyVoice TTS 后端。

    将 CosyVoice 的文本前端（:class:`CosyVoiceTokenizer`）、Flow Matching
    声学模型（:class:`FlowMatchingModel`）、HiFi-GAN 声码器
    （:class:`HiFiGanVocoder`）、语音 Tokenizer（:class:`SpeechTokenizer`）
    和说话人编码器（:class:`SpeakerEncoder`）组装为统一的
    :class:`TTSBackend`。

    与自回归后端的关键区别：
    * ``acoustic_type="flow_matching"``（非 ``"ar"``）
    * :meth:`synthesize` 内部通过 ODE 求解生成 mel，再经 HiFi-GAN 解码
    * :meth:`synthesize_stream` 使用 Chunk-aware ODE 求解（非逐 token 流式）

    Examples
    --------
    >>> backend = CosyVoiceBackend(model_path="/data/cosyvoice")
    >>> backend.load(device="cuda", dtype="float16")
    >>> audio = backend.synthesize("你好，世界", speaker=None, language="zh")

    Notes
    -----
    CosyVoice 模型遵循 **Apache-2.0** 许可。输出采样率为 24000Hz
    （与 HuggingFace 仓库 ``FunAudioLLM/CosyVoice2-0.5B`` 的
    ``cosyvoice2.yaml`` 一致）。
    """

    name: str = "cosyvoice"
    spec: TTSBackendSpec = TTSBackendSpec(
        name="cosyvoice",
        supported_languages=["zh", "en", "ja", "ko", "yue", "de", "fr"],
        supports_streaming=True,
        supports_voice_clone=True,
        vocoder_type="hifi_gan",
        acoustic_type="flow_matching",
        min_gpu_memory_gb=4.0,
        model_license="Apache-2.0",
        sample_rate=24000,
        default_params={
            "num_ode_steps": 10,
            "ode_solver": "euler",
            "speed": 1.0,
        },
    )

    def __init__(
        self,
        model_path: str,
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        speech_tokenizer_model: str | None = None,
        speaker_encoder_model: str = "campp",
        hifi_gan_path: str | None = None,
        num_ode_steps: int = 10,
        ode_solver: str = "euler",
        language: str = "zh",
        streaming_enabled: bool = True,
        chunk_size_frames: int = 150,
        chunk_overlap_frames: int = 16,
        scheduler: Any = None,
        repo_id: str | None = None,
    ) -> None:
        """初始化 CosyVoice 后端。

        Parameters
        ----------
        model_path : str
            CosyVoice 模型根路径。本地不存在时，将通过
            :class:`HFModelManager` 从 ``repo_id`` / 默认仓库下载。
        llm_model : str
            文本理解 LLM 模型名称或路径（HF 布局未找到 ``CosyVoice-BlankEN``
            时的回退）。
        speech_tokenizer_model : str | None
            语音 Tokenizer 权重路径；``None`` 时按 HF 布局查找
            ``speech_tokenizer_v2.onnx``，找不到再回退到旧目录
            ``model_path/speech_tokenizer``。
        speaker_encoder_model : str
            说话人编码器类型，默认 ``"campp"``。
        hifi_gan_path : str | None
            HiFi-GAN 权重路径；``None`` 时按 HF 布局查找 ``hift.pt``，
            找不到再回退到旧目录 ``model_path/hifi_gan``。
        num_ode_steps : int
            ODE 求解步数，默认 ``10``。
        ode_solver : str
            ODE 求解器，``"euler"`` / ``"midpoint"`` / ``"rk4"``。
        language : str
            默认语言代码。
        streaming_enabled : bool
            是否启用流式合成。
        chunk_size_frames : int
            流式 chunk 大小（帧数），默认 ``150``。
        chunk_overlap_frames : int
            chunk 重叠帧数，默认 ``16``。
        scheduler : Any
            显存调度器实例。
        repo_id : str | None
            HuggingFace 仓库 ID（如 ``"FunAudioLLM/CosyVoice2-0.5B"``）。
            ``None`` 时使用后端默认仓库 ``cosyvoice``。
        """
        super().__init__(scheduler=scheduler)

        self._model_path: str = model_path
        self._llm_model_name: str = llm_model
        # 用户显式提供的路径覆盖；None 表示在 _build_pipeline 中按 HF 布局解析
        self._speech_tokenizer_path: str | None = speech_tokenizer_model
        self._speaker_encoder_model: str = speaker_encoder_model
        self._hifi_gan_path: str | None = hifi_gan_path
        self._repo_id: str | None = repo_id
        self._num_ode_steps: int = num_ode_steps
        self._ode_solver: str = ode_solver
        self._language: str = language
        self._streaming_enabled: bool = streaming_enabled
        self._chunk_size_frames: int = chunk_size_frames
        self._chunk_overlap_frames: int = chunk_overlap_frames

        # LLM 模型实例
        self._llm: Any = None
        # E4-6: LLM hidden_dim -> FlowMatching condition_dim 投影层（按需创建）
        self._projection: Any = None
        self._flow_condition_dim: int = 512
        # 语音 Tokenizer 实例
        self._speech_tokenizer: Any = None
        # 说话人编码器实例
        self._speaker_encoder: Any = None
        # 说话人特征缓存
        self._speaker_cache: dict[str, dict[str, Any]] = {}
        self._speaker_dir: str = os.path.join(model_path, "speaker")

        # mel 帧率（帧/秒）
        self._mel_fps: float = 86.13

    # ==================================================================
    # 生命周期：组装 / 销毁管线
    # ==================================================================
    def _build_pipeline(self) -> None:
        """组装 CosyVoice 管线。

        依次构建并加载：

        1. Layer 1 — :class:`CosyVoiceTokenizer`（文本前端 + LLM tokenizer）
        2. LLM 模型（文本理解，如 Qwen2.5）
        3. Layer 2 — :class:`FlowMatchingModel`（Flow Matching 声学模型）
        4. Layer 3 — :class:`HiFiGanVocoder`（HiFi-GAN 声码器，复用已有代码）
        5. :class:`SpeechTokenizer`（参考音频编码）
        6. :class:`SpeakerEncoder`（说话人嵌入提取）
        7. Layer 4 — :class:`StreamAdapter`（流式适配，按需构建）
        8. 加载预计算的说话人嵌入
        """
        from mosaic.nodes.audio.tts_backends.acoustic_models.flow_matching import (
            FlowMatchingModel,
        )
        from mosaic.nodes.audio.tts_backends.acoustic_models.speaker_encoder import (
            SpeakerEncoder,
        )
        from mosaic.nodes.audio.tts_backends.acoustic_models.speech_tokenizer import (
            SpeechTokenizer,
        )
        from mosaic.nodes.audio.tts_backends.hf_model_manager import HFModelManager
        from mosaic.nodes.audio.tts_backends.streaming.base import StreamAdapter
        from mosaic.nodes.audio.tts_backends.text_frontends.cosyvoice_tokenizer import (
            CosyVoiceTokenizer,
        )
        from mosaic.nodes.audio.tts_backends.vocoders.hifi_gan import (
            HiFiGanVocoder,
        )

        # ------------------------------------------------------------------
        # 0. 确保模型已下载 & 读取 cosyvoice2.yaml 配置
        # ------------------------------------------------------------------
        model_dir = HFModelManager.ensure_model(
            self._model_path,
            repo_id=self._repo_id,
            backend_name="cosyvoice",
        )
        self._logger.info("CosyVoice model directory: %s", model_dir)

        yaml_config = HFModelManager.load_yaml_config(
            os.path.join(model_dir, "cosyvoice2.yaml")
        )
        sample_rate = int(yaml_config.get("sample_rate", self.spec.sample_rate))
        spk_embed_dim = int(yaml_config.get("spk_embed_dim", 192))
        llm_input_size = int(yaml_config.get("llm_input_size", 896))
        speech_token_size = int(yaml_config.get("speech_token_size", 6561))
        self._logger.info(
            "CosyVoice yaml config: sample_rate=%d spk_embed_dim=%d "
            "llm_input_size=%d speech_token_size=%d",
            sample_rate, spk_embed_dim, llm_input_size, speech_token_size,
        )

        # ------------------------------------------------------------------
        # 路径解析：HuggingFace 仓库布局优先，向后兼容旧目录布局
        # ------------------------------------------------------------------
        # LLM tokenizer 目录（Qwen2 模型目录，含 config.json/tokenizer）
        llm_tokenizer_dir = HFModelManager.find_dir(
            model_dir, ["CosyVoice-BlankEN", "llm", "tokenizer"]
        )
        if not llm_tokenizer_dir:
            # 向后兼容：旧 model_path/llm 目录
            llm_tokenizer_dir = os.path.join(model_dir, "llm")
        if not os.path.isdir(llm_tokenizer_dir):
            llm_tokenizer_dir = self._llm_model_name
        # LLM 权重（CosyVoice 自有 LLM checkpoint，如 llm.pt）
        llm_weight_path = HFModelManager.find_file(
            model_dir, ["llm.pt", "llm.safetensors"]
        )
        self._logger.info("LLM tokenizer path: %s", llm_tokenizer_dir)
        self._logger.info(
            "LLM weights: %s", llm_weight_path or "(use tokenizer dir)"
        )

        # Flow Matching 权重
        flow_weight_path = HFModelManager.find_file(
            model_dir, ["flow.pt", "flow.safetensors", "flow_matching.safetensors"]
        )
        if not flow_weight_path:
            # 向后兼容：旧 model_path/flow_matching 目录
            legacy_flow = os.path.join(model_dir, "flow_matching")
            flow_weight_path = legacy_flow
        self._logger.info("Flow Matching weights: %s", flow_weight_path)

        # HiFi-GAN 声码器权重（用户显式覆盖优先）
        if self._hifi_gan_path and (
            os.path.isfile(self._hifi_gan_path)
            or os.path.isdir(self._hifi_gan_path)
        ):
            hifi_gan_weight_path = self._hifi_gan_path
        else:
            hifi_gan_weight_path = HFModelManager.find_file(
                model_dir, ["hift.pt", "hifi_gan.safetensors", "vocoder.pt"]
            )
            if not hifi_gan_weight_path:
                # 向后兼容：旧 model_path/hifi_gan 目录
                hifi_gan_weight_path = os.path.join(model_dir, "hifi_gan")
        self._logger.info("HiFi-GAN vocoder weights: %s", hifi_gan_weight_path)

        # Speech tokenizer（用户显式覆盖优先）
        if self._speech_tokenizer_path and (
            os.path.isfile(self._speech_tokenizer_path)
            or os.path.isdir(self._speech_tokenizer_path)
        ):
            speech_tokenizer_weight_path = self._speech_tokenizer_path
        else:
            speech_tokenizer_weight_path = HFModelManager.find_file(
                model_dir,
                [
                    "speech_tokenizer_v2.onnx",
                    "speech_tokenizer_v2.batch.onnx",
                    "speech_tokenizer.onnx",
                ],
            )
            if not speech_tokenizer_weight_path:
                # 向后兼容：旧 model_path/speech_tokenizer 目录
                speech_tokenizer_weight_path = os.path.join(
                    model_dir, "speech_tokenizer"
                )
        self._logger.info(
            "Speech tokenizer weights: %s", speech_tokenizer_weight_path
        )

        # Speaker encoder 权重
        speaker_encoder_weight_path = HFModelManager.find_file(
            model_dir,
            ["campplus.onnx", "speaker_encoder.onnx", "speaker_encoder.pt"],
        )
        if not speaker_encoder_weight_path:
            # 向后兼容：旧 model_path/speaker_encoder 目录
            legacy_spk = os.path.join(model_dir, "speaker_encoder")
            speaker_encoder_weight_path = legacy_spk
        self._logger.info(
            "Speaker encoder weights: %s", speaker_encoder_weight_path
        )

        # ------------------------------------------------------------------
        # Layer 1: 文本前端 —— CosyVoiceTokenizer + LLM
        # ------------------------------------------------------------------
        self._text_frontend = CosyVoiceTokenizer(
            llm_model_path=llm_tokenizer_dir,
        )
        try:
            self._text_frontend.load_weights(
                weights_path=llm_tokenizer_dir,
                device=self._device,
                dtype=self._dtype,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load LLM tokenizer from %s: %s. "
                "Using dummy tokenizer.",
                llm_tokenizer_dir,
                exc,
            )

        # 加载 LLM 模型（文本理解）
        self._load_llm(llm_tokenizer_dir)

        # ------------------------------------------------------------------
        # Layer 2: Flow Matching 声学模型
        # ------------------------------------------------------------------
        self._acoustic_model = FlowMatchingModel(
            model_path=flow_weight_path,
            llm_model_path=llm_tokenizer_dir,
            in_channels=80,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            condition_dim=512,
            num_ode_steps=self._num_ode_steps,
            ode_solver=self._ode_solver,
        )
        self._acoustic_model.load_weights(
            weights_path=flow_weight_path,
            device=self._device,
            dtype=self._dtype,
        )

        # E4-6: 检查 LLM hidden_dim 与 FlowMatching condition_dim 是否匹配，
        # 不匹配时创建投影层（LLM 与 Flow 均已加载，可安全读取维度）
        self._setup_llm_projection()

        # ------------------------------------------------------------------
        # Layer 3: HiFi-GAN 声码器（复用已有代码）
        # ------------------------------------------------------------------
        self._vocoder = HiFiGanVocoder(
            model_path=hifi_gan_weight_path,
            sample_rate=sample_rate,
        )
        self._vocoder.load_weights(
            weights_path=hifi_gan_weight_path,
            device=self._device,
            dtype=self._dtype,
        )

        # ------------------------------------------------------------------
        # SpeechTokenizer + SpeakerEncoder
        # ------------------------------------------------------------------
        self._speech_tokenizer = SpeechTokenizer(
            model_path=speech_tokenizer_weight_path,
        )
        try:
            self._speech_tokenizer.load_weights(
                weights_path=speech_tokenizer_weight_path,
                device=self._device,
                dtype=self._dtype,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load SpeechTokenizer: %s. "
                "Voice clone will be limited.",
                exc,
            )

        self._speaker_encoder = SpeakerEncoder(
            model_type=self._speaker_encoder_model,
            embedding_dim=spk_embed_dim,
        )
        try:
            self._speaker_encoder.load_weights(
                weights_path=speaker_encoder_weight_path,
                device=self._device,
                dtype=self._dtype,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load SpeakerEncoder: %s.", exc
            )

        # ------------------------------------------------------------------
        # Layer 4: 流式适配器
        # ------------------------------------------------------------------
        if self._streaming_enabled:
            self._stream_adapter = StreamAdapter(
                chunk_size=4096,
                overlap=256,
                sample_rate=sample_rate,
            )

        # ------------------------------------------------------------------
        # 加载预计算的说话人嵌入
        # ------------------------------------------------------------------
        self._load_speaker_cache()

    def _destroy_pipeline(self) -> None:
        """销毁管线并释放资源。"""
        super()._destroy_pipeline()

        if self._llm is not None:
            try:
                if hasattr(self._llm, "to"):
                    self._llm.to("cpu")
            except Exception:  # noqa: BLE001
                pass
        self._llm = None

        if self._speech_tokenizer is not None:
            try:
                self._speech_tokenizer.unload_weights()
            except Exception:  # noqa: BLE001
                pass
        self._speech_tokenizer = None

        if self._speaker_encoder is not None:
            try:
                self._speaker_encoder.unload_weights()
            except Exception:  # noqa: BLE001
                pass
        self._speaker_encoder = None

        self._speaker_cache.clear()

    def _load_llm(self, llm_path: str) -> None:
        """加载 LLM 模型用于文本理解。

        Parameters
        ----------
        llm_path : str
            LLM 模型路径。
        """
        try:
            import torch  # type: ignore  # noqa: F401
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(llm_path)
            self._llm = AutoModel.from_pretrained(llm_path)
            self._llm.eval()
            self._llm_tokenizer = tokenizer

            resolved = self._device
            if resolved.startswith("cuda"):
                import torch

                if not torch.cuda.is_available():
                    resolved = "cpu"
            self._llm = self._llm.to(resolved)

            self._logger.info(
                "LLM loaded from %s for text understanding.", llm_path
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load LLM from %s: %s. "
                "Text understanding will use tokenizer-only mode.",
                llm_path,
                exc,
            )
            self._llm = None

    def _setup_llm_projection(self) -> None:
        """检查并创建 LLM→Flow 维度投影层（E4-6）。

        CosyVoice 的 LLM（如 Qwen2）``last_hidden_state`` 维度通常为
        ``896``，而 :class:`FlowMatchingModel` 的 ``condition_dim`` 为
        ``512``。维度不一致时，``text_feats`` 直接喂给 FlowMatchingModel
        会在内部 ``text_proj``（``Linear(cond, cond)``）处因形状不匹配而
        报错。此处按需创建一个 ``nn.Linear(llm_hidden, flow_cond)`` 投影层。

        .. warning::

           该投影层为随机初始化（无对应预训练权重），仅作为维度对齐的最小
           修复，不保证最优音质；完整方案应加载 CosyVoice 官方
           ``text_proj`` 权重。
        """
        self._projection = None
        if self._llm is None or self._acoustic_model is None:
            return
        # 读取 LLM hidden_size（防御性：部分模型可能没有 config.hidden_size）
        llm_hidden_dim = None
        try:
            llm_hidden_dim = getattr(
                getattr(self._llm, "config", None), "hidden_size", None
            )
        except Exception:  # noqa: BLE001
            llm_hidden_dim = None
        # 读取 Flow condition_dim（私有属性，回退到 512）
        flow_condition_dim = getattr(
            self._acoustic_model, "_condition_dim", 512
        )
        self._flow_condition_dim = flow_condition_dim
        if llm_hidden_dim is None:
            self._logger.info(
                "Cannot determine LLM hidden_size; skipping projection "
                "setup (assuming dims already match)."
            )
            return
        if llm_hidden_dim == flow_condition_dim:
            self._logger.info(
                "LLM hidden_dim (%d) == Flow condition_dim (%d); "
                "no projection needed.",
                llm_hidden_dim,
                flow_condition_dim,
            )
            return
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            self._projection = nn.Linear(llm_hidden_dim, flow_condition_dim)
            resolved = self._device
            if resolved.startswith("cuda") and not torch.cuda.is_available():
                resolved = "cpu"
            self._projection = self._projection.to(resolved)
            self._projection.eval()
            self._logger.info(
                "LLM hidden_dim (%d) != Flow condition_dim (%d); "
                "added projection layer (randomly initialized; load "
                "official text_proj weights for best quality).",
                llm_hidden_dim,
                flow_condition_dim,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to create LLM->Flow projection (%d->%d): %s. "
                "text_feats may have wrong dimension.",
                llm_hidden_dim,
                flow_condition_dim,
                exc,
            )
            self._projection = None

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

        1. 文本校验。
        2. CosyVoiceTokenizer.preprocess(text)
        3. token_ids = CosyVoiceTokenizer.tokenize(text, language)
        4. text_feats = LLM(token_ids) → text_hidden_states
        5. 参考音频处理（如果有 speaker）
        6. mel = FlowMatchingModel.generate(text_feats, speaker_info, ...)
        7. waveform = HiFiGanVocoder.decode(mel)
        8. 如果 speed != 1.0，做时间拉伸
        9. 返回 AudioData(waveform, sample_rate=24000)

        Parameters
        ----------
        text : str
            待合成文本。
        speaker : str | None
            说话人名称或参考音频路径。
        language : str
            语言代码。
        speed : float
            语速倍率。
        **kwargs : Any
            额外参数（``num_ode_steps`` / ``ode_solver`` 等）。

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

        # 2-3. 文本前端处理
        processed = self._text_frontend.preprocess(text)
        token_ids = self._text_frontend.tokenize(processed, language=language)

        # 4. LLM 文本理解 → text_feats
        text_feats = self._encode_text(token_ids)

        # 5. 参考音频处理
        speaker_info = self._get_speaker_info(speaker)

        # 6. 合并推理参数
        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        # 7. Flow Matching ODE 求解 → mel
        mel = self._acoustic_model.generate(
            token_ids=text_feats,
            speaker_embedding=speaker_info,
            **params,
        )

        # 8. HiFi-GAN 解码 → waveform
        waveform, sample_rate = self._decode_mel(mel)

        # 9. 语速调整
        if speed != 1.0 and speed > 0:
            waveform = self._adjust_speed(waveform, speed)

        # 10. 构造 AudioData
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
            "acoustic_type": "flow_matching",
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
        """分块流式合成。

        使用 Chunk-aware ODE 求解策略：
        1. 将目标 mel 分为多个 chunk（每个 ~150 帧 ≈ 1.5 秒）
        2. 每个 chunk 独立做 ODE 求解
        3. 每个 chunk 完成后经 HiFi-GAN 解码为波形
        4. 通过 StreamAdapter 缓冲输出

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
                text, speaker=speaker, language=language, speed=speed, **kwargs
            )
            return

        self._logger.info(
            "synthesize_stream: backend=%s language=%s speaker=%s "
            "speed=%.2f chunk_size=%d text_len=%d",
            self.name, language, speaker, speed, chunk_size, len(text),
        )

        # 文本前端处理
        processed = self._text_frontend.preprocess(text)
        token_ids = self._text_frontend.tokenize(processed, language=language)
        text_feats = self._encode_text(token_ids)

        # 参考音频处理
        speaker_info = self._get_speaker_info(speaker)

        # 合并参数
        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        # 估算目标 mel 帧数（文本长度 × 每字帧数）
        target_len = max(
            self._chunk_size_frames,
            int(len(text) * 15),  # 粗略估算：每字 ~15 帧
        )

        # 创建流式会话
        session = self._get_stream_session(chunk_size)

        try:
            # Chunk-aware ODE 求解 → 逐块解码
            for mel_chunk in self._acoustic_model.generate_stream(
                token_ids=text_feats,
                speaker_embedding=speaker_info,
                target_length=target_len,
                chunk_size_frames=self._chunk_size_frames,
                overlap_frames=self._chunk_overlap_frames,
                **params,
            ):
                # 流式取消：提前终止生成循环
                if session.is_cancelled is True:
                    self._logger.info(
                        "synthesize_stream cancelled for backend %s",
                        self.name,
                    )
                    break
                waveform_chunk, _sr = self._decode_mel(mel_chunk)
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

        Returns
        -------
        AudioData
            克隆语音结果。
        """
        self._ensure_loaded()
        speaker_info = self.extract_speaker(audio)
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
        """流式语音克隆。"""
        self._ensure_loaded()
        speaker_info = self.extract_speaker(audio)
        yield from self._synthesize_stream_with_speaker_info(
            text, speaker_info, language, **kwargs
        )

    # ==================================================================
    # 说话人管理
    # ==================================================================
    def list_speakers(self) -> list[str]:
        """返回预计算的说话人列表。"""
        return list(self._speaker_cache.keys())

    def extract_speaker(
        self, audio: AudioData | str
    ) -> dict[str, Any]:
        """从音频中提取说话人特征。

        Parameters
        ----------
        audio : AudioData | str
            参考音频。

        Returns
        -------
        dict[str, Any]
            包含 ``ref_speech_tokens`` 和 ``speaker_embedding`` 的字典。
        """
        self._ensure_loaded()

        waveform = self._get_waveform(audio)
        if waveform is None:
            return {
                "ref_speech_tokens": None,
                "speaker_embedding": None,
            }

        # SpeechTokenizer → ref_speech_tokens
        ref_speech_tokens = None
        if self._speech_tokenizer is not None:
            try:
                ref_speech_tokens = self._speech_tokenizer.encode(waveform)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("SpeechTokenizer encode failed: %s", exc)

        # SpeakerEncoder → speaker_embedding
        speaker_embedding = None
        if self._speaker_encoder is not None:
            try:
                speaker_embedding = self._speaker_encoder.encode(waveform)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("SpeakerEncoder encode failed: %s", exc)

        return {
            "ref_speech_tokens": ref_speech_tokens,
            "speaker_embedding": speaker_embedding,
        }

    def save_speaker(self, name: str, audio: AudioData | str) -> None:
        """提取并保存说话人特征到本地。"""
        self._ensure_loaded()
        speaker_info = self.extract_speaker(audio)
        self._speaker_cache[name] = speaker_info

        os.makedirs(self._speaker_dir, exist_ok=True)
        embeddings_path = os.path.join(self._speaker_dir, "embeddings.json")

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
            self._logger.info("Saved speaker %r to %s.", name, embeddings_path)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to save speaker: %s", exc)

    def load_speaker(self, name: str) -> dict[str, Any]:
        """加载已保存的说话人特征。"""
        if name not in self._speaker_cache:
            raise KeyError(f"Speaker {name!r} not found in cache.")
        return self._speaker_cache[name]

    # ==================================================================
    # ODE 参数管理
    # ==================================================================
    def set_ode_params(
        self, num_steps: int, solver: str = "euler"
    ) -> None:
        """运行时修改 ODE 参数。

        用于在质量和速度之间做权衡调整。

        Parameters
        ----------
        num_steps : int
            ODE 求解步数（5=最快, 10=推荐, 20=高质量）。
        solver : str
            ODE 求解器：``"euler"`` / ``"midpoint"`` / ``"rk4"``。
        """
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if solver not in ("euler", "midpoint", "rk4"):
            raise ValueError(
                f"Unsupported solver: {solver!r}. "
                f"Supported: 'euler', 'midpoint', 'rk4'."
            )
        self._num_ode_steps = num_steps
        self._ode_solver = solver
        if self._acoustic_model is not None:
            self._acoustic_model._num_ode_steps = num_steps
            self._acoustic_model._ode_solver = solver
        self._logger.info(
            "ODE params updated: steps=%d solver=%s", num_steps, solver
        )

    def benchmark_ode_steps(
        self, text: str, steps_list: list[int] | None = None
    ) -> dict[int, dict[str, float]]:
        """测试不同 ODE 步数的质量和速度。

        Parameters
        ----------
        text : str
            测试文本。
        steps_list : list[int] | None
            要测试的步数列表，默认 ``[5, 10, 20, 50]``。

        Returns
        -------
        dict[int, dict[str, float]]
            ``{步数: {"time": 耗时秒, "mel_std": mel标准差}}``。
        """
        self._ensure_loaded()
        if steps_list is None:
            steps_list = [5, 10, 20, 50]

        results: dict[int, dict[str, float]] = {}

        processed = self._text_frontend.preprocess(text)
        token_ids = self._text_frontend.tokenize(processed, language="zh")
        text_feats = self._encode_text(token_ids)

        for steps in steps_list:
            start = time.time()
            mel = self._acoustic_model.generate(
                token_ids=text_feats,
                speaker_embedding=None,
                num_ode_steps=steps,
            )
            elapsed = time.time() - start

            try:
                import torch

                mel_std = float(mel.std().item()) if hasattr(mel, "std") else 0.0
            except (ImportError, Exception):  # noqa: BLE001
                mel_std = 0.0

            results[steps] = {"time": elapsed, "mel_std": mel_std}
            self._logger.info(
                "benchmark: steps=%d time=%.3fs mel_std=%.4f",
                steps, elapsed, mel_std,
            )

        return results

    # ==================================================================
    # 依赖检查
    # ==================================================================
    @classmethod
    def check_dependencies(cls) -> bool:
        """检查运行时依赖是否可用。"""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    # ==================================================================
    # 内部辅助
    # ==================================================================
    def _encode_text(self, token_ids: Any) -> Any:
        """使用 LLM 编码文本 token ids 为特征。

        Parameters
        ----------
        token_ids : Any
            文本 token ids。

        Returns
        -------
        Any
            文本特征 ``[batch, seq_len, hidden]``。

        Raises
        ------
        RuntimeError
            LLM 未加载时抛出。E4-2：此前在 LLM 加载失败时静默返回原始
            ``token_ids``（整数 token id 张量）作为 ``text_feats`` 直接喂给
            FlowMatchingModel 当条件，维度/语义全错，产出垃圾音频却无任何
            错误抛出。现改为显式抛错，避免静默生成无效音频。
        """
        if self._llm is None:
            raise RuntimeError(
                "CosyVoice LLM model is not loaded. "
                "Cannot generate speech without LLM: text_feats would be "
                "raw integer token ids (wrong dimension/semantics for the "
                "FlowMatching condition). Please ensure the LLM is properly "
                "loaded (check model paths and the load() log for warnings)."
            )

        try:
            import torch

            with torch.no_grad():
                outputs = self._llm(token_ids)
                # 取最后一层隐藏状态
                if hasattr(outputs, "last_hidden_state"):
                    text_feats = outputs.last_hidden_state
                elif hasattr(outputs, "logits"):
                    text_feats = outputs.logits
                elif isinstance(outputs, tuple):
                    text_feats = outputs[0]
                else:
                    text_feats = outputs
                # E4-6: LLM hidden_dim 可能与 FlowMatching condition_dim 不一致
                # （如 Qwen2 hidden=896 vs Flow condition=512），按需投影对齐，
                # 否则 text_feats 喂给 FlowMatchingModel.text_proj 会形状报错。
                if self._projection is not None:
                    text_feats = self._projection(text_feats)
                return text_feats
        except Exception as exc:  # noqa: BLE001
            # E4-2：LLM 推理失败时同样不应静默回退到 token_ids（会产出垃圾音频）
            raise RuntimeError(
                f"CosyVoice LLM encoding failed: {exc}. Cannot produce valid "
                f"text features for the FlowMatching model."
            ) from exc

    def _decode_mel(self, mel: Any) -> tuple[Any, int]:
        """通过 HiFi-GAN 声码器将 mel 解码为波形。

        Parameters
        ----------
        mel : Any
            mel spectrogram。

        Returns
        -------
        tuple[Any, int]
            ``(waveform, sample_rate)``。
        """
        if self._vocoder is None:
            raise RuntimeError("Vocoder is not loaded.")
        result = self._vocoder.decode(mel)
        if isinstance(result, tuple):
            return result
        return (result, self.spec.sample_rate)

    def _get_speaker_info(
        self, speaker: str | None
    ) -> dict[str, Any] | None:
        """获取说话人信息。"""
        if speaker is None:
            return None

        if speaker in self._speaker_cache:
            return self._speaker_cache[speaker]

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

        # A3-4: 未知 speaker 记录警告，回退默认音色（避免静默）
        self._logger.warning(
            "Speaker %r not found in cache or as a file path; "
            "falling back to default speaker.",
            speaker,
        )
        return None

    def _get_waveform(self, audio: AudioData | str) -> Any:
        """从 AudioData 或文件路径获取波形。

        使用统一的参考音频预处理工具 :func:`load_reference_audio` 加载、
        校验时长并自动截断到 CosyVoice 推荐时长上限（10 秒），避免超长
        参考音频导致 OOM 或晦涩错误。注意 CosyVoice 模型采样率为
        24000Hz（非 22050Hz）。
        """
        is_audio_data = hasattr(audio, "waveform") and hasattr(
            audio, "sample_rate"
        )
        is_file = isinstance(audio, str) and os.path.isfile(audio)
        if not (is_audio_data or is_file):
            return None
        try:
            waveform, _sr = load_reference_audio(
                audio, target_sr=24000, backend="cosyvoice"
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
        text_feats = self._encode_text(token_ids)

        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        mel = self._acoustic_model.generate(
            token_ids=text_feats,
            speaker_embedding=speaker_info,
            **params,
        )

        waveform, sample_rate = self._decode_mel(mel)

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
                "acoustic_type": "flow_matching",
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
        text_feats = self._encode_text(token_ids)

        params: dict[str, Any] = dict(self.spec.default_params)
        params.update(kwargs)

        target_len = max(
            self._chunk_size_frames,
            int(len(text) * 15),
        )

        session = self._get_stream_session(kwargs.get("chunk_size", 4096))

        try:
            for mel_chunk in self._acoustic_model.generate_stream(
                token_ids=text_feats,
                speaker_embedding=speaker_info,
                target_length=target_len,
                chunk_size_frames=self._chunk_size_frames,
                overlap_frames=self._chunk_overlap_frames,
                **params,
            ):
                waveform_chunk, _sr = self._decode_mel(mel_chunk)
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
        """调整语速（时间拉伸）。"""
        try:
            import torch

            if isinstance(waveform, torch.Tensor):
                if waveform.dim() == 1:
                    n = waveform.shape[0]
                    new_n = max(1, int(n / speed))
                    indices = torch.linspace(
                        0, n - 1, new_n, device=waveform.device
                    ).long()
                    return waveform[indices]
                elif waveform.dim() == 2:
                    n = waveform.shape[-1]
                    new_n = max(1, int(n / speed))
                    indices = torch.linspace(
                        0, n - 1, new_n, device=waveform.device
                    ).long()
                    return waveform[..., indices]
        except ImportError:
            pass

        try:
            import numpy as np  # type: ignore

            if isinstance(waveform, np.ndarray):
                n = waveform.shape[-1]
                new_n = max(1, int(n / speed))
                indices = np.linspace(0, n - 1, new_n).astype(int)
                return waveform[..., indices]
        except ImportError:
            pass

        return waveform
