# mosaic/nodes/audio/music_generator.py
"""MusicGenerator 节点 —— 音乐生成。

根据文字描述生成音乐，基于 Facebook MusicGen 模型。
支持 small/medium/large 三个版本，默认使用 small 以节省显存。

设计要点
--------
* 使用 ``transformers`` 的 ``MusicgenForConditionalGeneration`` 加载模型。
* MusicGen 最大生成时长约 30 秒，超出时给出警告。
* 输出采样率统一处理为标准值（MusicGen 默认 32000Hz）。
* 生成的音频经过归一化处理。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData

from mosaic.nodes.audio._base import BaseAudioNode

__all__ = ["MusicGenerator"]


@registry.register
class MusicGenerator(BaseAudioNode):
    """音乐生成节点。

    根据文字描述生成音乐片段，基于 Facebook MusicGen。

    Parameters
    ----------
    model:
        MusicGen 模型标识，默认 ``"facebook/musicgen-small"``。
        可选 ``"facebook/musicgen-medium"`` 或 ``"facebook/musicgen-large"``。
    **kwargs:
        透传给 :class:`BaseAudioNode` 的参数。

    Examples
    --------
    >>> music = MusicGenerator()
    >>> result = music(MosaicData(
    ...     prompt="轻松的钢琴曲，适合冥想",
    ...     duration=10.0,
    ... ))
    >>> audio = result["audio"]  # AudioData
    """

    name: str = "music-generator"
    description: str = (
        "Generate music from text descriptions using Facebook MusicGen. "
        "Supports duration control and guidance scale."
    )
    version: str = "0.1.0"
    input_types = ["text", "mosaic"]
    output_types = ["audio"]

    def __init__(
        self,
        model: str = "facebook/musicgen-small",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)

    def _load_model(self) -> None:
        """加载 MusicGen 模型。"""
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoProcessor,
            MusicgenForConditionalGeneration,
        )
        from mosaic.nodes._pipeline_utils import (
            safe_load_processor,
            safe_load_model,
        )

        device = self._resolve_device()
        try:
            resolved_dtype = torch.float16 if "cuda" in device else torch.float32
        except (AttributeError, RuntimeError):
            resolved_dtype = torch.float32

        self._processor = safe_load_processor(AutoProcessor, self._model_name)
        self._model = safe_load_model(
            MusicgenForConditionalGeneration,
            self._model_name,
            dtype=resolved_dtype,
        )
        self._model.to(device)

        self._logger.info(
            "MusicGen model loaded (model=%s, device=%s).",
            self._model_name,
            device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行音乐生成。

        Parameters
        ----------
        input_data:
            必须包含 ``prompt`` (str)；可选 ``duration`` (float, 默认 8.0)、
            ``guidance_scale`` (float, 默认 3.0)。

        Returns
        -------
        MosaicData
            包含 ``audio`` (AudioData)、``prompt`` (str)、``duration`` (float)。

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
                    f"MusicGenerator requires 'prompt' (non-empty str), "
                    f"got {type(prompt).__name__}."
                )

            duration = float(input_data.get("duration", 8.0))
            guidance_scale = float(input_data.get("guidance_scale", 3.0))

            # MusicGen 最大生成时长约 30 秒
            if duration > 30.0:
                self._logger.warning(
                    "MusicGen max duration is ~30s, requested %.1fs. "
                    "Truncating to 30s.",
                    duration,
                )
                duration = 30.0

            # MusicGen 默认采样率
            sample_rate = 32000
            max_new_tokens = int(duration * sample_rate / 512 / 2)
            max_new_tokens = max(1, max_new_tokens)

            # 编码输入
            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )

            # 迁移到设备
            device = self._infer_device()
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except (AttributeError, RuntimeError):
                pass

            # 生成
            import torch  # type: ignore

            with torch.inference_mode():
                audio_values = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    guidance_scale=guidance_scale,
                    do_sample=True,
                )

            # 提取波形（转 float32 避免 float16 输出）
            waveform = audio_values[0, 0].cpu().numpy()
            # 显式转 float32，避免 float16 透传到导出环节
            import numpy as np

            if isinstance(waveform, np.ndarray) and waveform.dtype == np.float16:
                waveform = waveform.astype(np.float32)
            actual_duration = self._get_duration(waveform, sample_rate)
        except Exception as exc:
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
            duration=actual_duration,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "duration": actual_duration,
                "sample_rate": sample_rate,
                "guidance_scale": guidance_scale,
            },
        )
        return result
