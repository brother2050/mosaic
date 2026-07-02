# mosaic/nodes/consistency/_base.py
"""一致性域节点基类。

提取 :class:`IdentityKeeper`、:class:`StyleKeeper`、
:class:`CrossFrameConsistency` 共用的图像前后处理逻辑与事件发射辅助。

设计要点
--------
* 复用图像域基类的惰性导入模式（``diffusers`` / ``torch`` 在实际
  加载时才导入），使本模块在依赖缺失时仍可被注册表发现。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理。
* 一致性域是**生成控制域**，节点通常与图像域或视频域的生成节点
  组合使用，在生成过程中注入一致性约束。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。
"""

from __future__ import annotations

import abc
import logging
import random
from typing import Any

from mosaic.core.device_utils import (
    apply_optimizations,
    auto_resolve_device_dtype,
    infer_device,
    resolve_device,
    resolve_dtype,
    run_diffusers_pipeline,
)
from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData

__all__ = ["BaseConsistencyNode"]


# 常见一致性模型的粗略显存估算（fp16，GB）
_VRAM_ESTIMATES: dict[str, float] = {
    "InstantX/InstantID": 12.0,
    "h94/IP-Adapter": 6.0,
    "h94/IP-Adapter-SDXL": 8.0,
    "TencentARC/PhotoMaker": 12.0,
    "stabilityai/stable-diffusion-xl-base-1.0": 8.0,
}

# 许可证信息
_LICENSE_INFO: dict[str, str] = {
    "InstantX/InstantID": "Apache-2.0 (model weights)",
    "h94/IP-Adapter": "Apache-2.0",
    "h94/IP-Adapter-SDXL": "Apache-2.0",
    "TencentARC/PhotoMaker": "Apache-2.0 (model weights)",
    "stabilityai/stable-diffusion-xl-base-1.0": "OpenRAIL++-M",
}


class BaseConsistencyNode(Node):
    """一致性域节点抽象基类。

    提供图像前后处理工具方法和事件发射辅助。子类需实现
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

    domain: str = "consistency"
    description: str = "Base consistency node."
    version: str = "0.1.0"
    input_types: tuple[str, ...] = ("image", "mosaic")
    output_types: tuple[str, ...] = ("image",)

    #: 图像尺寸上限（单边像素），超过则抛出 ValueError 以避免内存溢出。
    MAX_DIMENSION: int = 4096

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "float16",
        scheduler: Scheduler | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        model_name = kwargs.pop("model", "")
        super().__init__(bus=bus, **kwargs)
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._logger = logging.getLogger(
            f"mosaic.nodes.consistency.{self.name}"
        )

        # 自动解析设备与 dtype：CPU/SD1.5 环境下将 float16 降级为 float32
        self._device, self._dtype_str = auto_resolve_device_dtype(
            device, dtype, self._scheduler, self._logger,
            model_name=model_name,
        )

        # 运行时持有的 Pipeline / 模型
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # 模型加载辅助
    # ------------------------------------------------------------------
    def _resolve_dtype(self) -> Any:
        """解析 torch dtype 字符串为 torch.dtype 对象。"""
        return resolve_dtype(self._dtype_str)

    def _infer_device(self) -> str:
        """推断推理设备。"""
        if self._pipeline is None:
            return self._scheduler.device
        # 优先从 pipeline 的 UNet 参数推断设备，失败时由工具函数回退到调度器设备
        unet = getattr(self._pipeline, "unet", None)
        return infer_device(
            unet if unet is not None else self._pipeline, self._scheduler
        )

    def _resolve_device(self) -> str:
        """解析实际推理设备，无 GPU 时从调度器降级并记录日志。

        统一的设备解析入口：当节点配置为 CUDA 但调度器检测不到 GPU 时，
        降级到调度器报告的设备（通常为 ``"cpu"``）并记录日志。各域子类应
        通过本方法解析目标设备，避免各自实现不一致的降级逻辑。
        """
        resolved = resolve_device(self._device, self._scheduler)
        if resolved != self._device:
            self._logger.info(
                "Device resolved: %s → %s", self._device, resolved,
            )
        return resolved

    def _check_image_dimensions(self, width: int, height: int) -> None:
        """校验图像尺寸是否在合理上限内，避免大图像导致内存溢出。

        Parameters
        ----------
        width, height:
            待校验的图像宽 / 高（像素）。

        Raises
        ------
        ValueError
            ``width`` 或 ``height`` 超过 :attr:`MAX_DIMENSION`。
        """
        if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
            raise ValueError(
                f"Image dimensions too large: {width}x{height}, "
                f"max {self.MAX_DIMENSION}x{self.MAX_DIMENSION}"
            )

    def _prepare_seed(self, seed: int | None) -> tuple[int, Any]:
        """准备随机种子与 generator。

        Parameters
        ----------
        seed:
            用户指定的种子，``None`` 表示随机。

        Returns
        -------
        tuple[int, torch.Generator | None]
            ``(actual_seed, generator)``。
        """
        import torch  # type: ignore

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed) % (2**32)

        device = self._infer_device()
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        except (RuntimeError, ValueError, TypeError):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        return seed, generator

    def _run_pipeline(self, **kwargs: Any) -> Any:
        """在 torch.inference_mode 下执行 Pipeline 调用。"""
        return run_diffusers_pipeline(self._pipeline, **kwargs)

    def _apply_optimizations(self) -> None:
        """对已加载的 Pipeline 应用显存优化配置。"""
        apply_optimizations(
            self._pipeline,
            enable_attention_slicing=True,
            enable_vae_slicing=True,
        )

    # ------------------------------------------------------------------
    # 图像前后处理工具
    # ------------------------------------------------------------------
    @staticmethod
    def _load_image(source: Any) -> Any:
        """从文件路径或 PIL.Image 加载图像。

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
            if not __import__("os").path.exists(source):
                raise FileNotFoundError(f"Image file not found: {source}")
            return Image.open(source).convert("RGB")
        raise TypeError(
            f"Expected str (file path) or PIL.Image.Image, "
            f"got {type(source).__name__}."
        )

    @staticmethod
    def _resize_to_model(
        image: Any,
        target_size: tuple[int, int] = (512, 512),
    ) -> Any:
        """将图像 resize 到模型要求尺寸。

        Parameters
        ----------
        image:
            ``PIL.Image.Image`` 实例。
        target_size:
            目标尺寸 ``(width, height)``，默认 ``(512, 512)``。

        Returns
        -------
        PIL.Image.Image
            resize 后的图像。
        """
        from PIL import Image  # type: ignore

        if not isinstance(image, Image.Image):
            image = BaseConsistencyNode._load_image(image)

        w, h = target_size
        # 对齐到 8 的倍数
        w = max(8, (w // 8) * 8)
        h = max(8, (h // 8) * 8)

        if (w, h) != image.size:
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _prepare_face_region(
        image: Any,
        padding_ratio: float = 0.2,
    ) -> tuple[Any, tuple[int, int, int, int]]:
        """提取人脸区域。

        使用 ``insightface`` 进行人脸检测（如果可用），否则使用简单的
        中心裁剪作为回退方案。

        Parameters
        ----------
        image:
            ``PIL.Image.Image`` 实例。
        padding_ratio:
            人脸框向外扩展的比例，默认 ``0.2``（20%）。

        Returns
        -------
        tuple[PIL.Image.Image, tuple[int, int, int, int]]
            ``(face_image, bbox)``，bbox 为 ``(x1, y1, x2, y2)``。

        Raises
        ------
        ValueError
            参考图中检测不到人脸。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if not isinstance(image, Image.Image):
            image = BaseConsistencyNode._load_image(image)

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
                    "No face detected in the reference image. "
                    "Please provide an image with a clear, visible face."
                )

            # 取最大的人脸
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            x1, y1, x2, y2 = face.bbox.astype(int)

            # 添加 padding
            fw = x2 - x1
            fh = y2 - y1
            px = int(fw * padding_ratio)
            py = int(fh * padding_ratio)
            x1 = max(0, x1 - px)
            y1 = max(0, y1 - py)
            x2 = min(w, x2 + px)
            y2 = min(h, y2 + py)

            face_image = image.crop((x1, y1, x2, y2))
            return face_image, (x1, y1, x2, y2)

        except ImportError:
            pass

        # 回退：中心裁剪（假设人脸在中心）
        cx, cy = w // 2, h // 2
        crop_size = min(w, h) // 2
        x1 = max(0, cx - crop_size)
        y1 = max(0, cy - crop_size)
        x2 = min(w, cx + crop_size)
        y2 = min(h, cy + crop_size)
        face_image = image.crop((x1, y1, x2, y2))
        return face_image, (x1, y1, x2, y2)

    @staticmethod
    def _compute_image_similarity(img1: Any, img2: Any) -> float:
        """计算两张图片的结构相似度。

        使用简单的像素级 SSIM（结构相似性指数）进行计算。如果
        ``scikit-image`` 可用，则使用其实现；否则使用降采样的直方图
        相关作为回退方案。

        Parameters
        ----------
        img1:
            第一张图片（``PIL.Image.Image``）。
        img2:
            第二张图片（``PIL.Image.Image``）。

        Returns
        -------
        float
            相似度分数（0.0-1.0），1.0 表示完全相同。
        """
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if not isinstance(img1, Image.Image):
            img1 = BaseConsistencyNode._load_image(img1)
        if not isinstance(img2, Image.Image):
            img2 = BaseConsistencyNode._load_image(img2)

        # 统一尺寸
        target = (256, 256)
        img1 = img1.resize(target, Image.Resampling.BILINEAR)
        img2 = img2.resize(target, Image.Resampling.BILINEAR)

        arr1 = np.array(img1.convert("RGB"), dtype=np.float32) / 255.0
        arr2 = np.array(img2.convert("RGB"), dtype=np.float32) / 255.0

        # 尝试使用 scikit-image 的 SSIM
        try:
            from skimage.metrics import structural_similarity as ssim  # type: ignore

            score = ssim(arr1, arr2, channel_axis=2, data_range=1.0)
            return float(max(0.0, min(1.0, score)))
        except ImportError:
            pass

        # 回退：归一化互相关
        arr1_flat = arr1.flatten()
        arr2_flat = arr2.flatten()
        arr1_flat = arr1_flat - arr1_flat.mean()
        arr2_flat = arr2_flat - arr2_flat.mean()

        denom = (
            np.linalg.norm(arr1_flat) * np.linalg.norm(arr2_flat) + 1e-8
        )
        correlation = float(np.dot(arr1_flat, arr2_flat) / denom)
        return max(0.0, min(1.0, (correlation + 1.0) / 2.0))

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

    def _build_model_info(self, model_name: str) -> dict[str, Any]:
        """构造模型信息字典。"""
        vram = _VRAM_ESTIMATES.get(model_name, 10.0)
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
