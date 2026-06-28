# mosaic/nodes/digital_human/lip_syncer.py
"""LipSyncer 节点 —— 口型同步。

根据一张人脸图片与一段音频，生成口型与音频内容同步的数字人说话视频。
仅修改嘴部区域，其余部分（眼睛、鼻子、背景等）保持不变。

设计要点
--------
* 继承 :class:`BaseDigitalHumanNode`，复用人脸检测、裁剪、面部融合等
  工具方法。
* 三种后端模型（``method``）：
    - ``musetalk``：MuseTalk UNet 口型驱动 + wav2vec 音频特征，显存约 6-8GB；
    - ``wav2lip``：Wav2Lip 生成器 + 判别器，显存约 2-4GB；
    - ``sadtalker``：SadTalker 口型模块，显存约 4-6GB。
* 音频时长决定总帧数 ``total_frames = duration * fps``；若输入是单张图片，
  则复制多份以匹配音频时长；若输入是帧序列/视频，则按需循环填充。
* 逐帧处理：裁剪人脸区域（带 ``padding`` 扩展）送入模型，仅将模型输出的
  嘴部子区域融合回原图，保证其他部分不变。
* 所有第三方库（torch / diffusers / transformers / imageio / PIL / numpy）
  惰性导入，使本模块在依赖缺失时仍可被注册表发现与导入。
* 关键步骤通过事件总线发出 start / progress / complete / error 事件。
* 模型生命周期由 :class:`~mosaic.core.scheduler.Scheduler` 管理。

显存需求
--------
* ``KwaiVGI/MuseTalk``：约 6-8GB（fp16）
* ``wav2lip``：约 2-4GB（fp16）
* ``cvitkwai/SadTalker``：约 4-6GB（fp16）

许可证
------
* MuseTalk：CC-BY-NC 4.0
* Wav2Lip：CC-BY-NC 4.0 (research only)
* SadTalker：Apache-2.0 (code), CC-BY-NC-4.0 (model weights)

Limitations
-----------
* 真正的高质量口型同步依赖对应模型库（musetalk / wav2lip / sadtalker），
  未安装时 ``load`` 会抛出带安装提示的 ``ImportError``。
* 人脸图片需包含清晰可检测的人脸；检测不到人脸时抛出 ``ValueError``。
* ``parsing_mode="jaw"`` 使用关键点估算嘴部区域；非 jaw 模式回退到嘴部
  中心扩展框。
* 音频过短（< 0.1s）时仅生成 1 帧；音频过长会逐帧处理，耗时线性增长。
* 建议人脸在画面中占比较大，过小的人脸会降低口型同步质量。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import AudioData, MosaicData, VideoData

from mosaic.nodes.digital_human._base import BaseDigitalHumanNode

__all__ = ["LipSyncer"]


# method -> 默认模型标识
_DEFAULT_MODELS: Dict[str, str] = {
    "musetalk": "KwaiVGI/MuseTalk",
    "wav2lip": "wav2lip",
    "sadtalker": "cvitkwai/SadTalker",
}

# method -> 粗略显存需求（GB, fp16），用于 describe()
_METHOD_VRAM: Dict[str, float] = {
    "musetalk": 7.0,
    "wav2lip": 3.0,
    "sadtalker": 5.0,
}

# method -> 许可证（覆盖未知模型的默认值）
_METHOD_LICENSE: Dict[str, str] = {
    "musetalk": "CC-BY-NC 4.0",
    "wav2lip": "CC-BY-NC 4.0 (research only)",
    "sadtalker": "Apache-2.0 (code), CC-BY-NC-4.0 (model weights)",
}

# method -> 人脸裁剪工作尺寸
_WORK_SIZE: Dict[str, Tuple[int, int]] = {
    "musetalk": (256, 256),
    "wav2lip": (96, 96),
    "sadtalker": (256, 256),
}

_DEFAULT_FPS: int = 25
_DEFAULT_PADDING: List[int] = [0, 20, 0, 20]  # [left, top, right, bottom]


@registry.register
class LipSyncer(BaseDigitalHumanNode):
    """口型同步节点。

    根据人脸图片与音频生成口型同步视频，仅修改嘴部区域。

    Parameters
    ----------
    model:
        HuggingFace 模型标识或本地路径，默认 ``"KwaiVGI/MuseTalk"``。
        当 ``method`` 改变而 ``model`` 仍为默认值时，自动按 ``method`` 解析
        对应默认模型；显式指定其他 ``model`` 时以用户指定为准。
    method:
        口型同步方法，可选 ``"musetalk"`` / ``"wav2lip"`` / ``"sadtalker"``，
        默认 ``"musetalk"``。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，默认 ``"float16"``。
    **kwargs:
        透传给 :class:`BaseDigitalHumanNode` 的参数（如 ``scheduler``/``bus``）。

    Examples
    --------
    >>> syncer = LipSyncer(method="musetalk")
    >>> result = syncer(MosaicData(
    ...     face_image="face.jpg",
    ...     audio="speech.wav",
    ...     fps=25,
    ... ))
    >>> video = result["video"]  # VideoData
    """

    name: str = "lip-syncer"
    description: str = (
        "Lip-sync a face image to an audio track. Supports MuseTalk, Wav2Lip "
        "and SadTalker backends; only the mouth region is modified."
    )
    version: str = "0.1.0"
    input_types: List[str] = ["image", "audio", "video", "mosaic"]
    output_types: List[str] = ["video", "image", "audio", "mosaic"]

    def __init__(
        self,
        model: str = "KwaiVGI/MuseTalk",
        method: str = "musetalk",
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self._method: str = (method or "musetalk").lower().strip()
        # 以 method 为主导解析默认模型；用户显式指定其他 model 时保留之
        method_default = _DEFAULT_MODELS.get(self._method)
        if method_default is not None and (
            model is None or model == "KwaiVGI/MuseTalk"
        ):
            self._model_name: str = method_default
        else:
            self._model_name = model

        # 运行时子模块引用（load 后填充）
        self._discriminator: Any = None
        self._audio_encoder: Any = None

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载口型同步模型到 GPU/CPU。

        根据 :attr:`_method` 分发到对应的加载逻辑，加载完成后调用
        :meth:`_apply_optimizations` 应用显存优化。
        """
        self._scheduler.track(self)

        if self._pipeline is not None or self._model is not None:
            self._loaded = True
            return

        self._logger.info(
            "Loading lip-syncer model %s (method=%s) ...",
            self._model_name,
            self._method,
        )

        if self._method == "musetalk":
            self._load_musetalk()
        elif self._method == "wav2lip":
            self._load_wav2lip()
        elif self._method == "sadtalker":
            self._load_sadtalker()
        else:
            raise ValueError(
                f"Unsupported method {self._method!r}. "
                f"Expected one of: {sorted(_DEFAULT_MODELS)}."
            )

        self._apply_optimizations()
        self._loaded = True
        self._logger.info(
            "Lip-syncer loaded (method=%s, device=%s, dtype=%s).",
            self._method,
            self._resolve_device(),
            self._dtype_str,
        )

    def _load_musetalk(self) -> None:
        """加载 MuseTalk 模型（UNet 口型驱动 + wav2vec 音频特征）。

        显存约 6-8GB（fp16）。
        """
        import os
        import torch  # type: ignore

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        try:
            from musetalk import MuseTalk  # type: ignore

            self._model = MuseTalk.from_pretrained(
                self._model_name, torch_dtype=dtype
            )
            self._model = self._model.to(device)
        except ImportError:
            self._logger.warning(
                "'musetalk' package not found. Attempting structured "
                "checkpoint loading from %s.",
                self._model_name,
            )
            if not os.path.isdir(self._model_name):
                raise ImportError(
                    "MuseTalk backend requires the 'musetalk' package or a "
                    "local checkpoint directory. Install via "
                    "`pip install musetalk` or set `model` to a local path."
                )
            unet_ckpt = os.path.join(self._model_name, "unet.pth")
            if not os.path.exists(unet_ckpt):
                raise FileNotFoundError(
                    f"MuseTalk UNet checkpoint not found at {unet_ckpt!r}."
                )
            self._model = {
                "unet": torch.load(unet_ckpt, map_location=device),
            }

        # wav2vec 音频特征提取器
        try:
            from transformers import (  # type: ignore
                Wav2Vec2Model,
                Wav2Vec2Processor,
            )

            self._processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self._audio_encoder = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h"
            ).to(device)
        except ImportError:
            self._logger.debug(
                "transformers not available; wav2vec audio encoder disabled."
            )

    def _load_wav2lip(self) -> None:
        """加载 Wav2Lip 模型（生成器 + 判别器）。

        Wav2Lip 由一个生成器网络（人脸口型合成）与一个判别器（同步性评估）
        组成。显存约 2-4GB（fp16）。
        """
        import os
        import torch  # type: ignore

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        try:
            from wav2lip import Wav2Lip  # type: ignore

            self._model = Wav2Lip.from_pretrained(
                self._model_name, torch_dtype=dtype
            )
            self._model = self._model.to(device)
            # 判别器（用于同步性评估，推理可选但按需加载）
            try:
                self._discriminator = self._model.get_discriminator()
            except Exception:  # noqa: BLE001
                self._discriminator = None
        except ImportError:
            self._logger.warning(
                "'wav2lip' package not found. Attempting structured "
                "checkpoint loading from %s.",
                self._model_name,
            )
            if not os.path.isdir(self._model_name):
                raise ImportError(
                    "Wav2Lip backend requires the 'wav2lip' package or a "
                    "local checkpoint directory. Install via "
                    "`pip install wav2lip` or set `model` to a local path."
                )
            gen_ckpt = os.path.join(self._model_name, "wav2lip.pth")
            if not os.path.exists(gen_ckpt):
                raise FileNotFoundError(
                    f"Wav2Lip generator checkpoint not found at {gen_ckpt!r}."
                )
            self._model = {
                "generator": torch.load(gen_ckpt, map_location=device),
            }
            disc_ckpt = os.path.join(self._model_name, "wav2lip-disc.pth")
            if os.path.exists(disc_ckpt):
                self._discriminator = torch.load(disc_ckpt, map_location=device)

    def _load_sadtalker(self) -> None:
        """加载 SadTalker 口型模块。

        SadTalker 基于 3DMM 建模，这里加载其口型/表情驱动子模块，
        显存约 4-6GB（fp16）。
        """
        import os
        import torch  # type: ignore

        device = self._resolve_device()
        dtype = self._resolve_dtype()

        try:
            from sadtalker import SadTalker  # type: ignore

            self._model = SadTalker(
                checkpoint_path=self._model_name,
                device=device,
                dtype=dtype,
            )
        except ImportError:
            self._logger.warning(
                "'sadtalker' package not found. Attempting structured "
                "checkpoint loading from %s.",
                self._model_name,
            )
            if not os.path.isdir(self._model_name):
                raise ImportError(
                    "SadTalker backend requires the 'sadtalker' package or a "
                    "local checkpoint directory. Install via "
                    "`pip install sadtalker` or set `model` to a local path."
                )
            components: Dict[str, Any] = {}
            for name in ("mapping", "generator", "kp_extractor", "renderer"):
                ckpt = os.path.join(self._model_name, f"{name}.pth")
                if os.path.exists(ckpt):
                    components[name] = torch.load(ckpt, map_location=device)
            if not components:
                raise FileNotFoundError(
                    f"No SadTalker checkpoints found under {self._model_name!r}."
                )
            self._model = components

    def unload(self) -> None:
        """释放口型同步模型与显存。"""
        self._pipeline = None
        self._model = None
        self._processor = None
        self._discriminator = None
        self._audio_encoder = None
        self._loaded = False
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self._logger.info("Lip-syncer unloaded (method=%s).", self._method)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行口型同步。

        Parameters
        ----------
        input_data:
            必须包含 ``face_image`` (PIL.Image | str | VideoData |
            List[PIL.Image]) 与 ``audio`` (AudioData | str | ndarray)。
            可选：``fps`` (int, 默认 25)、
            ``output_format`` ("video"|"frames", 默认 "video")、
            ``padding`` ([left, top, right, bottom], 默认 [0,20,0,20])、
            ``parsing_mode`` ("jaw"|"default", 默认 "jaw")。

        Returns
        -------
        MosaicData
            含 ``video`` (VideoData) 或 ``frames`` (List[PIL.Image])、
            ``audio`` (AudioData)、``duration``、``fps``。

        Raises
        ------
        ValueError
            缺少 ``face_image`` 或 ``audio``、检测不到人脸、method 不支持。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 1. 校验输入
            face_input = input_data.get("face_image")
            if face_input is None:
                raise ValueError(
                    "LipSyncer requires 'face_image' "
                    "(PIL.Image, file path, VideoData, or list of frames)."
                )
            audio_input = input_data.get("audio")
            if audio_input is None:
                raise ValueError(
                    "LipSyncer requires 'audio' "
                    "(AudioData, file path, or ndarray)."
                )

            fps = int(input_data.get("fps", _DEFAULT_FPS))
            output_format = str(
                input_data.get("output_format", "video")
            ).lower()
            padding = self._parse_padding(
                input_data.get("padding", _DEFAULT_PADDING)
            )
            parsing_mode = str(
                input_data.get("parsing_mode", "jaw")
            ).lower()

            # 2. 加载音频并计算总帧数
            waveform, sample_rate = self._load_audio_signal(audio_input)
            duration = self._get_audio_duration(waveform, sample_rate)
            total_frames = max(1, int(round(duration * fps)))
            self._logger.info(
                "Audio duration=%.2fs, total_frames=%d (fps=%d).",
                duration,
                total_frames,
                fps,
            )

            # 3. 准备基础帧（单张图片复制 / 帧序列循环填充）
            base_frames = self._prepare_base_frames(face_input, total_frames)

            # 4. 检测人脸并计算嘴部区域
            first_frame = base_frames[0]
            _, bbox, landmarks = self._detect_face(first_frame)
            padded_bbox = self._expand_bbox(bbox, padding, first_frame.size)
            mouth_bbox = self._mouth_bbox(
                landmarks, padded_bbox, first_frame.size, parsing_mode
            )
            self._logger.info(
                "Face bbox=%s, padded=%s, mouth=%s.",
                bbox,
                padded_bbox,
                mouth_bbox,
            )

            # 5. 逐帧口型同步
            frames = self._run_lip_sync(
                base_frames,
                padded_bbox,
                mouth_bbox,
                waveform,
                sample_rate,
                fps,
                total_frames,
                parsing_mode,
            )
        except Exception as exc:
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0
        out_duration = len(frames) / fps if fps > 0 and frames else 0.0

        # 包装输出音频
        out_audio = AudioData(
            waveform=waveform,
            sample_rate=sample_rate,
            metadata={"duration": duration, "format": "wav"},
        )

        result = MosaicData(
            audio=out_audio,
            duration=out_duration,
            fps=fps,
        )
        if output_format == "frames":
            result["frames"] = frames
        else:
            width, height = (frames[0].size if frames else (0, 0))
            result["video"] = VideoData(
                frames=frames,
                fps=fps,
                metadata={
                    "duration": out_duration,
                    "width": width,
                    "height": height,
                    "frame_count": len(frames),
                    "method": self._method,
                    "parsing_mode": parsing_mode,
                },
            )

        self._emit_complete(
            duration=elapsed,
            output_summary={
                "method": self._method,
                "frame_count": len(frames),
                "duration": out_duration,
                "fps": fps,
                "parsing_mode": parsing_mode,
            },
        )
        return result

    # ------------------------------------------------------------------
    # 口型同步主流程
    # ------------------------------------------------------------------
    def _run_lip_sync(
        self,
        base_frames: List[Any],
        padded_bbox: Tuple[int, int, int, int],
        mouth_bbox: Tuple[int, int, int, int],
        waveform: Any,
        sample_rate: int,
        fps: int,
        total_frames: int,
        parsing_mode: str,
    ) -> List[Any]:
        """逐帧执行口型同步，仅融合嘴部区域回原图。"""
        work_size = _WORK_SIZE.get(self._method, (256, 256))

        self._emit_progress(0, total_frames, "Lip syncing")

        out_frames: List[Any] = []
        for i in range(total_frames):
            base_frame = base_frames[i % len(base_frames)]

            # 裁剪人脸区域（带 padding 扩展）并 resize 到工作尺寸
            face_crop = self._crop_and_resize(
                base_frame, padded_bbox, target_size=work_size
            )

            # 模型推理：生成口型同步后的人脸
            modified_face = self._sync_mouth(
                face_crop,
                waveform,
                sample_rate,
                frame_idx=i,
                fps=fps,
                parsing_mode=parsing_mode,
            )

            # 仅将嘴部子区域融合回原图
            frame = self._blend_mouth_only(
                base_frame,
                modified_face,
                padded_bbox,
                mouth_bbox,
                work_size,
            )
            out_frames.append(frame)
            self._emit_progress(
                i + 1, total_frames, f"Lip-synced frame {i + 1}/{total_frames}"
            )

        return out_frames

    def _sync_mouth(
        self,
        face_crop: Any,
        waveform: Any,
        sample_rate: int,
        frame_idx: int,
        fps: int,
        parsing_mode: str,
    ) -> Any:
        """调用模型对单帧人脸做口型同步，返回修改后的人脸（工作尺寸）。

        根据 :attr:`_method` 分发到对应模型调用。
        """
        if self._method == "musetalk":
            output = self._model(
                face_image=face_crop,
                audio=waveform,
                sample_rate=sample_rate,
                frame_index=frame_idx,
                fps=fps,
            )
        elif self._method == "wav2lip":
            output = self._model(
                face_image=face_crop,
                audio=waveform,
                sample_rate=sample_rate,
                frame_index=frame_idx,
            )
        else:  # sadtalker
            output = self._model.lip_forward(
                face_image=face_crop,
                audio=waveform,
                sample_rate=sample_rate,
                frame_index=frame_idx,
            )

        images = self._extract_images(output)
        return images[0] if images else face_crop

    def _blend_mouth_only(
        self,
        base_frame: Any,
        modified_face: Any,
        padded_bbox: Tuple[int, int, int, int],
        mouth_bbox: Tuple[int, int, int, int],
        work_size: Tuple[int, int],
    ) -> Any:
        """将模型输出中的嘴部子区域融合回原图，其他部分保持不变。

        模型输出 ``modified_face`` 为工作尺寸的人脸（仅嘴部改变）。本方法
        将 ``mouth_bbox``（原图坐标）映射到工作尺寸坐标系，从模型输出中
        裁剪嘴部子区域，再融合回原图的 ``mouth_bbox`` 位置。
        """
        from PIL import Image  # type: ignore

        if not isinstance(modified_face, Image.Image):
            modified_face = Image.fromarray(modified_face)

        px1, py1, px2, py2 = padded_bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)
        ww, wh = work_size
        sx = ww / pw
        sy = wh / ph

        mx1, my1, mx2, my2 = mouth_bbox
        # 嘴部区域在工作尺寸坐标系中的位置
        cmx1 = max(0, int((mx1 - px1) * sx))
        cmy1 = max(0, int((my1 - py1) * sy))
        cmx2 = min(ww, int((mx2 - px1) * sx))
        cmy2 = min(wh, int((my2 - py1) * sy))

        if cmx2 <= cmx1 or cmy2 <= cmy1:
            # 嘴部区域无法映射，回退到整张人脸融合
            return self._blend_face(
                base_frame, modified_face, padded_bbox, blend_ratio=1.0
            )

        mouth_face = modified_face.crop((cmx1, cmy1, cmx2, cmy2))
        return self._blend_face(
            base_frame, mouth_face, mouth_bbox, blend_ratio=1.0
        )

    # ------------------------------------------------------------------
    # 信号加载与几何工具
    # ------------------------------------------------------------------
    def _load_audio_signal(self, audio: Any) -> Tuple[Any, int]:
        """加载音频为 (waveform, sample_rate)。"""
        from mosaic.nodes.audio._base import BaseAudioNode

        return BaseAudioNode._load_audio(audio)

    @staticmethod
    def _get_audio_duration(waveform: Any, sample_rate: int) -> float:
        """计算音频时长（秒）。"""
        from mosaic.nodes.audio._base import BaseAudioNode

        return BaseAudioNode._get_duration(waveform, sample_rate)

    def _prepare_base_frames(
        self, face_input: Any, total_frames: int
    ) -> List[Any]:
        """准备基础帧序列。

        * 单张图片（str / PIL.Image）：复制 ``total_frames`` 份；
        * 帧序列（VideoData / list）：按需循环填充到 ``total_frames``。
        """
        from PIL import Image  # type: ignore

        if isinstance(face_input, (str, Image.Image)):
            img = self._load_image(face_input)
            return [img.copy() for _ in range(total_frames)]

        if isinstance(face_input, VideoData):
            frames = face_input.frames
        elif isinstance(face_input, list):
            frames = face_input
        else:
            raise TypeError(
                f"face_image must be PIL.Image, str, VideoData, or list, "
                f"got {type(face_input).__name__}."
            )

        if not frames:
            raise ValueError("face_image contains no frames.")

        out: List[Any] = []
        for i in range(total_frames):
            f = frames[i % len(frames)]
            if not isinstance(f, Image.Image):
                f = Image.fromarray(f)
            out.append(f)
        return out

    @staticmethod
    def _parse_padding(padding: Any) -> Tuple[int, int, int, int]:
        """解析 padding 为 (left, top, right, bottom)。"""
        if isinstance(padding, (int, float)):
            p = int(padding)
            return (p, p, p, p)
        if isinstance(padding, (list, tuple)):
            vals = [int(v) for v in padding]
            if len(vals) >= 4:
                return (vals[0], vals[1], vals[2], vals[3])
            if len(vals) == 1:
                return (vals[0], vals[0], vals[0], vals[0])
        return (0, 20, 0, 20)

    @staticmethod
    def _expand_bbox(
        bbox: Tuple[int, int, int, int],
        padding: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """按 [left, top, right, bottom] 扩展 bbox 并裁剪到图像范围。"""
        x1, y1, x2, y2 = bbox
        pad_l, pad_t, pad_r, pad_b = padding
        w, h = image_size
        x1 = max(0, x1 - pad_l)
        y1 = max(0, y1 - pad_t)
        x2 = min(w, x2 + pad_r)
        y2 = min(h, y2 + pad_b)
        return (x1, y1, x2, y2)

    @staticmethod
    def _mouth_bbox(
        landmarks: Any,
        face_bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        parsing_mode: str,
    ) -> Tuple[int, int, int, int]:
        """根据关键点估算嘴部区域 ``(x1, y1, x2, y2)``。

        ``parsing_mode="jaw"`` 时使用嘴部关键点 + 下颌扩展；否则使用嘴部
        中心扩展框。
        """
        import numpy as np  # type: ignore

        x1, y1, x2, y2 = face_bbox
        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)
        w, h = image_size

        if landmarks is None or len(landmarks) < 5:
            # 回退：人脸下 1/3 区域
            mx1 = x1
            mx2 = x2
            my1 = y1 + int(fh * 0.60)
            my2 = y2
            return (
                max(0, mx1),
                max(0, my1),
                min(w, mx2),
                min(h, my2),
            )

        lm = np.asarray(landmarks, dtype=np.float32)
        left_mouth = lm[3]
        right_mouth = lm[4]
        mouth_cx = float((left_mouth[0] + right_mouth[0]) / 2.0)
        mouth_cy = float((left_mouth[1] + right_mouth[1]) / 2.0)
        mouth_w = float(np.linalg.norm(right_mouth - left_mouth))

        if parsing_mode == "jaw":
            # 下颌模式：嘴部区域向下扩展更多以包含下巴
            half_w = mouth_w * 0.85
            top = mouth_w * 0.35
            bottom = mouth_w * 1.05
        else:
            half_w = mouth_w * 0.75
            top = mouth_w * 0.40
            bottom = mouth_w * 0.60

        mx1 = int(mouth_cx - half_w)
        mx2 = int(mouth_cx + half_w)
        my1 = int(mouth_cy - top)
        my2 = int(mouth_cy + bottom)

        # 限制在人脸 bbox 范围内
        mx1 = max(x1, mx1)
        mx2 = min(x2, mx2)
        my1 = max(y1, my1)
        my2 = min(y2, my2)
        # 限制在图像范围内
        return (
            max(0, mx1),
            max(0, my1),
            min(w, mx2),
            min(h, my2),
        )

    @staticmethod
    def _extract_images(output: Any) -> List[Any]:
        """从模型输出中提取 PIL.Image 列表（兼容多种返回格式）。"""
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        raw: Any = None
        if isinstance(output, dict):
            for key in ("frames", "images", "image", "video", "frame"):
                if key in output:
                    raw = output[key]
                    break
        else:
            for attr in ("frames", "images", "image", "video", "frame"):
                raw = getattr(output, attr, None)
                if raw is not None:
                    break
        if raw is None:
            raw = output

        if hasattr(raw, "cpu"):
            from mosaic.nodes.video._base import BaseVideoNode

            return BaseVideoNode._tensor_to_frames(raw)

        if isinstance(raw, list):
            if raw and isinstance(raw[0], list):
                raw = raw[0]
            images: List[Any] = []
            for f in raw:
                if isinstance(f, Image.Image):
                    images.append(f)
                else:
                    images.append(Image.fromarray(np.asarray(f)))
            return images

        if isinstance(raw, Image.Image):
            return [raw]
        return [Image.fromarray(np.asarray(raw))]

    # ------------------------------------------------------------------
    # describe
    # ------------------------------------------------------------------
    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息（VRAM、许可证、method）。"""
        info = self._build_model_info(self._model_name)
        info["method"] = self._method
        # 用 method 粒度的显存估算与许可证覆盖（wav2lip 等未知模型更精确）
        if self._method in _METHOD_VRAM:
            info["vram_gb"] = _METHOD_VRAM[self._method]
        if self._method in _METHOD_LICENSE:
            info["license"] = _METHOD_LICENSE[self._method]
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=info,
        )
