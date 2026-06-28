# mosaic/nodes/digital_human/_base.py
"""数字人域节点基类。

提取 :class:`AvatarDriver`、:class:`LipSyncer`、
:class:`MotionGenerator`、:class:`RealtimeRenderer` 共用的人物图像
处理逻辑、模型加载辅助与事件发射方法。

设计要点
--------
* 复用一致性域基类的惰性导入模式（``torch`` / ``insightface`` /
  ``PIL`` 在实际加载时才导入），使本模块在依赖缺失时仍可被注册表发现。
* 数字人域是整个项目显存需求最高的域，所有节点通过
  :class:`~mosaic.core.scheduler.Scheduler` 管理显存。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。
* 人脸处理逻辑复用一致性域的 ``_prepare_face_region`` 模式，并扩展
  人脸对齐、特征提取、表情应用与面部融合等数字人专用操作。
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Tuple

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData

__all__ = ["BaseDigitalHumanNode"]


# 常见数字人模型的粗略显存估算（fp16，GB）
_VRAM_ESTIMATES: Dict[str, float] = {
    "KwaiVGI/LivePortrait": 5.0,
    "KwaiVGI/MuseTalk": 7.0,
    "cvitkwai/SadTalker": 5.0,
    "PrimeIntellect/MotionGPT": 6.0,
}

# 许可证信息
_LICENSE_INFO: Dict[str, str] = {
    "KwaiVGI/LivePortrait": "MIT License (model weights)",
    "KwaiVGI/MuseTalk": "CC-BY-NC 4.0",
    "cvitkwai/SadTalker": "Apache-2.0 (code), CC-BY-NC-4.0 (model weights)",
    "PrimeIntellect/MotionGPT": "Apache-2.0",
}


class BaseDigitalHumanNode(Node):
    """数字人域节点抽象基类。

    提供人物图像处理工具方法、模型加载辅助与事件发射方法。子类需实现
    :meth:`load`/:meth:`unload`/:meth:`run`/:meth:`describe`。

    Parameters
    ----------
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，可选 ``"float16"``/``"float32"``/``"bfloat16"``，
        默认 ``"float16"``。
    scheduler:
        显存调度器实例，``None`` 使用全局单例。
    bus:
        事件总线实例，``None`` 使用全局单例。
    """

    domain: str = "digital_human"
    description: str = "Base digital human node."
    version: str = "0.1.0"
    input_types: List[str] = ["image", "audio", "video", "text", "mosaic"]
    output_types: List[str] = ["video", "image", "mosaic"]

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float16",
        scheduler: Optional[Scheduler] = None,
        bus: Optional[EventBus] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._device: str = device
        self._dtype_str: str = dtype
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._bus: EventBus = bus or get_event_bus()
        self._logger = logging.getLogger(
            f"mosaic.nodes.digital_human.{self.name}"
        )

        # 运行时持有的 Pipeline / 模型
        self._pipeline: Any = None
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # 模型加载辅助
    # ------------------------------------------------------------------
    def _resolve_dtype(self) -> Any:
        """解析 torch dtype 字符串为 torch.dtype 对象。"""
        import torch  # type: ignore

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(self._dtype_str, torch.float16)

    def _resolve_device(self) -> str:
        """解析实际设备字符串，无 GPU 时降级到 CPU。"""
        try:
            import torch  # type: ignore

            if self._device.startswith("cuda") and not torch.cuda.is_available():
                self._logger.warning(
                    "CUDA not available, falling back to CPU for %s.",
                    self.name,
                )
                return "cpu"
        except ImportError:
            pass
        return self._device

    def _infer_device(self) -> str:
        """推断推理设备。"""
        if self._pipeline is None and self._model is None:
            return self._scheduler.device
        try:
            import torch  # type: ignore

            for obj in (self._pipeline, self._model):
                unet = getattr(obj, "unet", None)
                if unet is not None:
                    return next(unet.parameters()).device.type
                params = getattr(obj, "parameters", None)
                if params is not None:
                    return next(params()).device.type
        except (StopIteration, Exception):  # noqa: BLE001
            pass
        return self._device

    def _run_pipeline(self, **kwargs: Any) -> Any:
        """在 torch.inference_mode 下执行 Pipeline 调用。"""
        import torch  # type: ignore

        with torch.inference_mode():
            output = self._pipeline(**kwargs)
        return output

    def _apply_optimizations(self) -> None:
        """对已加载的 Pipeline 应用显存优化配置。"""
        pipe = self._pipeline
        if pipe is None:
            return
        try:
            pipe.enable_attention_slicing()
            self._logger.debug("Enabled attention slicing.")
        except Exception:  # noqa: BLE001
            pass
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_slicing"):
            try:
                vae.enable_slicing()
                self._logger.debug("Enabled VAE slicing.")
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                pipe.enable_vae_slicing()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 人物图像处理工具
    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(source: Any) -> Any:
        """从文件路径或 PIL.Image 加载图像（RGB 模式）。

        Parameters
        ----------
        source:
            文件路径 (str) 或 ``PIL.Image.Image`` 实例。

        Returns
        -------
        PIL.Image.Image
            加载后的 PIL 图像（RGB 模式）。

        Raises
        ------
        TypeError
            ``source`` 类型不支持。
        FileNotFoundError
            文件路径不存在。
        """
        from PIL import Image  # type: ignore

        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, str):
            import os

            if not os.path.exists(source):
                raise FileNotFoundError(f"Image file not found: {source}")
            return Image.open(source).convert("RGB")
        raise TypeError(
            f"Expected str (file path) or PIL.Image.Image, "
            f"got {type(source).__name__}."
        )

    @staticmethod
    def _detect_face(
        image: Any,
    ) -> Tuple[Any, Tuple[int, int, int, int], Any]:
        """检测人脸，返回人脸区域、边界框、关键点。

        使用 ``insightface`` 进行人脸检测（如果可用），否则使用简单
        的中心裁剪作为回退方案。

        Parameters
        ----------
        image:
            ``PIL.Image.Image`` 实例或文件路径。

        Returns
        -------
        Tuple[PIL.Image.Image, Tuple[int, int, int, int], Any]
            ``(face_image, bbox, landmarks)``，bbox 为 ``(x1, y1, x2, y2)``，
            landmarks 为 5 个关键点坐标（左眼、右眼、鼻尖、左嘴角、右嘴角）。

        Raises
        ------
        ValueError
            图像中检测不到人脸。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if not isinstance(image, Image.Image):
            image = BaseDigitalHumanNode._load_image(image)

        w, h = image.size

        # 尝试使用 insightface 检测人脸
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0, det_size=(640, 640))
            img_array = np.array(image)
            faces = app.get(img_array)

            if not faces:
                raise ValueError(
                    "No face detected in the image. "
                    "Please provide an image with a clear, visible face."
                )

            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0])
                * (f.bbox[3] - f.bbox[1]),
            )
            x1, y1, x2, y2 = face.bbox.astype(int)
            landmarks = getattr(face, "kps", None)

            face_image = image.crop((x1, y1, x2, y2))
            return face_image, (int(x1), int(y1), int(x2), int(y2)), landmarks

        except ImportError:
            pass

        # 回退：中心裁剪
        cx, cy = w // 2, h // 2
        crop_size = min(w, h) // 2
        x1 = max(0, cx - crop_size)
        y1 = max(0, cy - crop_size)
        x2 = min(w, cx + crop_size)
        y2 = min(h, cy + crop_size)
        face_image = image.crop((x1, y1, x2, y2))

        # 合成默认关键点（5 点）
        default_landmarks = np.array(
            [
                [x1 + (x2 - x1) * 0.35, y1 + (y2 - y1) * 0.40],  # 左眼
                [x1 + (x2 - x1) * 0.65, y1 + (y2 - y1) * 0.40],  # 右眼
                [x1 + (x2 - x1) * 0.50, y1 + (y2 - y1) * 0.55],  # 鼻尖
                [x1 + (x2 - x1) * 0.40, y1 + (y2 - y1) * 0.70],  # 左嘴角
                [x1 + (x2 - x1) * 0.60, y1 + (y2 - y1) * 0.70],  # 右嘴角
            ],
            dtype=np.float32,
        )
        return face_image, (x1, y1, x2, y2), default_landmarks

    @staticmethod
    def _extract_face_embedding(face_image: Any) -> Any:
        """提取人脸特征向量。

        使用 ``insightface`` 的 ArcFace 模型（如果可用），否则返回
        随机特征向量作为占位符。

        Parameters
        ----------
        face_image:
            ``PIL.Image.Image`` 人脸图像。

        Returns
        -------
        numpy.ndarray
            人脸特征向量，形状 ``(512,)``。
        """
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        if not isinstance(face_image, Image.Image):
            face_image = BaseDigitalHumanNode._load_image(face_image)

        # 尝试使用 insightface 提取特征
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0, det_size=(640, 640))
            img_array = np.array(face_image)
            faces = app.get(img_array)

            if faces:
                return faces[0].embedding  # shape (512,)
        except ImportError:
            pass

        # 回退：基于像素统计的简单特征
        arr = np.array(face_image.convert("L").resize((64, 64)), dtype=np.float32)
        return arr.flatten() / 255.0  # shape (4096,)

    @staticmethod
    def _align_face(
        image: Any,
        landmarks: Any,
        target_size: Tuple[int, int] = (256, 256),
    ) -> Any:
        """人脸对齐。

        根据关键点（左眼、右眼）计算旋转角度，对人脸进行仿射变换对齐。

        Parameters
        ----------
        image:
            ``PIL.Image.Image`` 实例。
        landmarks:
            关键点坐标数组，至少包含左眼和右眼位置。
        target_size:
            对齐后的人脸尺寸，默认 ``(256, 256)``。

        Returns
        -------
        PIL.Image.Image
            对齐后的人脸图像。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if not isinstance(image, Image.Image):
            image = BaseDigitalHumanNode._load_image(image)

        if landmarks is None or len(landmarks) < 2:
            return image.resize(target_size, Image.LANCZOS)

        landmarks = np.asarray(landmarks, dtype=np.float32)
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # 计算旋转角度
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # 以两眼中点为旋转中心
        center = (
            float((left_eye[0] + right_eye[0]) / 2),
            float((left_eye[1] + right_eye[1]) / 2),
        )

        # 旋转并对齐
        rotated = image.rotate(
            angle, center=center, resample=Image.BILINEAR
        )
        return rotated.resize(target_size, Image.LANCZOS)

    @staticmethod
    def _crop_and_resize(
        image: Any,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (256, 256),
        padding: int = 0,
    ) -> Any:
        """裁剪并调整大小。

        Parameters
        ----------
        image:
            ``PIL.Image.Image`` 实例。
        bbox:
            裁剪区域 ``(x1, y1, x2, y2)``。
        target_size:
            目标尺寸。
        padding:
            额外扩展像素数。

        Returns
        -------
        PIL.Image.Image
            裁剪并 resize 后的图像。
        """
        from PIL import Image  # type: ignore

        if not isinstance(image, Image.Image):
            image = BaseDigitalHumanNode._load_image(image)

        x1, y1, x2, y2 = bbox
        if padding > 0:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.size[0], x2 + padding)
            y2 = min(image.size[1], y2 + padding)

        cropped = image.crop((x1, y1, x2, y2))
        return cropped.resize(target_size, Image.LANCZOS)

    @staticmethod
    def _apply_expression(
        face_image: Any,
        expression_params: Dict[str, Any],
    ) -> Any:
        """应用表情参数到人脸图像。

        这是一个简化实现：通过调整图像亮度、对比度和色调来模拟表情变化。
        实际的表情驱动应由模型推理完成。

        Parameters
        ----------
        face_image:
            ``PIL.Image.Image`` 人脸图像。
        expression_params:
            表情参数字典，可包含：
            * ``smile`` (float, 0-1)：微笑程度
            * ``eye_openness`` (float, 0-1)：眼睛睁开程度
            * ``mouth_open`` (float, 0-1)：嘴巴张开程度
            * ``brow_raise`` (float, 0-1)：眉毛抬起程度

        Returns
        -------
        PIL.Image.Image
            应用了表情参数的人脸图像。
        """
        from PIL import Image, ImageEnhance  # type: ignore
        import numpy as np  # type: ignore

        if not isinstance(face_image, Image.Image):
            face_image = BaseDigitalHumanNode._load_image(face_image)

        smile = expression_params.get("smile", 0.0)
        mouth_open = expression_params.get("mouth_open", 0.0)

        # 简化实现：通过亮度/对比度调整模拟表情
        brightness = 1.0 + smile * 0.1
        contrast = 1.0 + mouth_open * 0.15

        enhancer = ImageEnhance.Brightness(face_image)
        face_image = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(face_image)
        face_image = enhancer.enhance(contrast)

        return face_image

    @staticmethod
    def _blend_face(
        original_image: Any,
        face_image: Any,
        bbox: Tuple[int, int, int, int],
        blend_ratio: float = 1.0,
    ) -> Any:
        """将生成的人脸融合回原图。

        使用 Poisson 融合或 alpha 混合将修改后的人脸区域无缝融合到
        原始图片中。

        Parameters
        ----------
        original_image:
            原始 ``PIL.Image.Image``。
        face_image:
            生成的人脸 ``PIL.Image.Image``。
        bbox:
            人脸区域 ``(x1, y1, x2, y2)``。
        blend_ratio:
            融合比例，``1.0`` 完全使用生成结果，``0.0`` 完全使用原图。

        Returns
        -------
        PIL.Image.Image
            融合后的完整图像。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if not isinstance(original_image, Image.Image):
            original_image = BaseDigitalHumanNode._load_image(original_image)

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return original_image

        # 将 face_image resize 到 bbox 大小
        resized_face = face_image.resize((w, h), Image.LANCZOS)

        # Alpha 混合
        result = original_image.copy()
        orig_region = np.array(
            original_image.crop((x1, y1, x2, y2)).convert("RGB"),
            dtype=np.float32,
        )
        face_region = np.array(resized_face.convert("RGB"), dtype=np.float32)

        blended = (
            orig_region * (1.0 - blend_ratio)
            + face_region * blend_ratio
        )
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        blended_image = Image.fromarray(blended)

        result.paste(blended_image, (x1, y1))
        return result

    # ------------------------------------------------------------------
    # 事件发射辅助
    # ------------------------------------------------------------------
    def _emit_start(self) -> None:
        """发出 node_start 事件。"""
        self._bus.emit(
            EventType.NODE_START,
            node_name=self.name,
            node_domain=self.domain,
        )

    def _emit_complete(self, duration: float, output_summary: Any) -> None:
        """发出 node_complete 事件。"""
        self._bus.emit(
            EventType.NODE_COMPLETE,
            node_name=self.name,
            duration=duration,
            output_summary=output_summary,
        )

    def _emit_error(self, error: BaseException) -> None:
        """发出 node_error 事件。"""
        self._bus.emit(
            EventType.NODE_ERROR,
            node_name=self.name,
            error=error,
        )

    def _emit_progress(
        self, current: int, total: int, message: str = ""
    ) -> None:
        """发出进度事件。"""
        self._bus.emit(
            EventType.PROGRESS,
            node_name=self.name,
            current=current,
            total=total,
            message=message,
        )

    # ------------------------------------------------------------------
    # Node 抽象方法
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load(self) -> None:
        """加载资源（子类实现）。"""

    @abc.abstractmethod
    def unload(self) -> None:
        """释放资源（子类实现）。"""

    @abc.abstractmethod
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行节点逻辑（子类实现）。"""

    @abc.abstractmethod
    def describe(self) -> NodeSpec:
        """返回节点规格说明（子类实现）。"""

    def _build_model_info(self, model_name: str) -> Dict[str, Any]:
        """构造模型信息字典。"""
        vram = _VRAM_ESTIMATES.get(model_name, 8.0)
        license_info = _LICENSE_INFO.get(
            model_name, "See model card on HuggingFace"
        )
        return {
            "name": model_name,
            "source": "HuggingFace",
            "license": license_info,
            "vram_gb": vram,
            "dtype": self._dtype_str,
            "device": self._device,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"domain={self.domain!r} state={status}>"
        )
