# mosaic/nodes/audio/sound_effect.py
"""SoundEffectGenerator 节点 —— 音效生成。

根据文字描述生成短音效，基于 AudioLDM2 模型。
适合生成 10 秒以内的短音效，如环境音、拟音、音效素材。

设计要点
--------
* 使用 ``diffusers.AudioLDMPipeline`` 加载 AudioLDM2 模型。
* 支持正向/反向提示词、推理步数控制。
* 输出重采样到标准采样率并归一化。
* AudioLDM2 适合短音效（< 10 秒），超长时给出警告。

Prompt 示例
-----------
好的 prompt 对生成质量影响很大，建议包含以下要素：
- 声音主体：``"下雨的声音"``、``"汽车喇叭"``、``"门铃声"``
- 环境描述：``"在空旷的走廊里"``、``"近距离录音"``
- 音质描述：``"清晰"``、``"有回声"``、``"低沉"``

示例：``"清脆的门铃声，近距离录音，无背景噪音"``
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData

from mosaic.nodes.audio._base import BaseAudioNode

__all__ = ["SoundEffectGenerator"]


@registry.register
class SoundEffectGenerator(BaseAudioNode):
    """音效生成节点。

    根据文字描述生成短音效，基于 AudioLDM2。

    Parameters
    ----------
    model:
        AudioLDM2 模型标识，默认 ``"cvssp/audioldm2"``。
    **kwargs:
        透传给 :class:`BaseAudioNode` 的参数。

    Examples
    --------
    >>> sfx = SoundEffectGenerator()
    >>> result = sfx(MosaicData(
    ...     prompt="下雨的声音，室内，轻柔的雨滴打在窗户上",
    ...     duration=5.0,
    ...     negative_prompt="嘈杂，刺耳",
    ... ))
    >>> audio = result["audio"]  # AudioData
    """

    name: str = "sound-effect-generator"
    description: str = (
        "Generate sound effects from text descriptions using AudioLDM2. "
        "Supports negative prompts and inference step control."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["audio"]

    def __init__(
        self,
        model: str = "cvssp/audioldm2",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)

    def _load_model(self) -> None:
        """加载 AudioLDM2 模型。"""
        import torch  # type: ignore
        from diffusers import AudioLDMPipeline  # type: ignore
        from mosaic.nodes._pipeline_utils import safe_load_pipeline

        device = self._resolve_device()
        try:
            torch_dtype = torch.float16 if "cuda" in device else torch.float32
        except (AttributeError, RuntimeError):
            torch_dtype = torch.float32

        # AudioLDM2 使用 T5 文本编码器，needs_t5=True 预导入 T5 组件
        self._pipeline = safe_load_pipeline(
            AudioLDMPipeline,
            self._model_name,
            needs_t5=True,
            torch_dtype=torch_dtype,
        )
        self._pipeline = self._pipeline.to(device)

        # 保存 pipeline 引用以便基类统一管理
        self._model = self._pipeline

        self._logger.info(
            "AudioLDM2 pipeline loaded (model=%s, device=%s).",
            self._model_name,
            device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行音效生成。

        Parameters
        ----------
        input_data:
            必须包含 ``prompt`` (str)；可选 ``duration`` (float, 默认 5.0)、
            ``negative_prompt`` (str)、``num_inference_steps`` (int, 默认 10)。

        Returns
        -------
        MosaicData
            包含 ``audio`` (AudioData)、``prompt`` (str)。

        Raises
        ------
        ValueError
            缺少 ``prompt`` 或 ``prompt`` 非字符串。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"SoundEffectGenerator requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}."
                )

            duration = float(input_data.get("duration", 5.0))
            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            num_inference_steps = int(input_data.get("num_inference_steps", 10))

            # AudioLDM2 适合短音效（< 10 秒）
            if duration > 10.0:
                self._logger.warning(
                    "AudioLDM2 is optimized for short effects (< 10s), "
                    "requested %.1fs. Quality may degrade.",
                    duration,
                )

            # AudioLDM2 默认采样率
            sample_rate = 16000
            # 音频长度（秒）通过 num_waveforms_per_prompt 控制不了，
            # 需要通过 audio_length_in_s 参数
            audio_length_in_s = min(duration, 10.0)

            # 构造 pipeline 参数
            pipe_kwargs: dict = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "audio_length_in_s": audio_length_in_s,
            }
            if negative_prompt is not None:
                pipe_kwargs["negative_prompt"] = negative_prompt

            # 执行推理
            import torch  # type: ignore

            with torch.inference_mode():
                output = self._pipeline(**pipe_kwargs)

            # 提取波形
            audios = output.audios if hasattr(output, "audios") else []
            if len(audios) > 0:
                import numpy as np  # type: ignore

                waveform = audios[0]
                # 显式转 float32，避免 float16 透传到导出环节
                if isinstance(waveform, np.ndarray):
                    waveform = waveform.astype(np.float32)
                    # 确保是 1D
                    if waveform.ndim > 1:
                        waveform = self._to_mono(waveform)
            else:
                import numpy as np  # type: ignore

                waveform = np.array([], dtype=np.float32)

            actual_duration = self._get_duration(waveform, sample_rate)
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # 包装为 AudioData
        audio = self._ensure_audio_data(
            waveform, sample_rate, prompt=prompt
        )

        result = MosaicData(
            audio=audio,
            prompt=prompt,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "duration": actual_duration,
                "sample_rate": sample_rate,
                "num_inference_steps": num_inference_steps,
            },
        )
        return result
