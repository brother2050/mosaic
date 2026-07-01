# mosaic/nodes/image/_base.py
"""图像域节点基类。

提取图像生成节点共用的 ``diffusers`` Pipeline 加载、推理与图像前后处理逻辑。
子类只需实现 :meth:`BaseImageNode.run` 中"如何构造 Pipeline 调用参数"与
"如何提取输出"的部分，底层推理流程由本基类提供。

设计要点
--------
* ``diffusers`` / ``torch`` 采用惰性导入，使本模块在未安装这些依赖时
  仍可被注册表发现与导入（仅在实际加载/推理时才报依赖缺失）。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理：
  ``load`` 调用 ``scheduler.track(self)`` 注册显存跟踪并执行实际加载；
  ``run`` 调用 ``scheduler.ensure_loaded(self)`` 触发按需加载 + LRU 淘汰。
  注意：``load`` 不能调用 ``ensure_loaded``（会递归）。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出 ``node_start``/
  ``node_complete``/``node_error`` 事件。
* 统一封装显存优化选项（attention_slicing / vae_slicing / CPU offload）。
"""

from __future__ import annotations

import abc
import logging
import random
from typing import Any

from mosaic.core._device_utils import (
    apply_optimizations,
    auto_resolve_device_dtype,
    infer_device,
    resolve_dtype,
    run_diffusers_pipeline,
    upcast_pipeline_components,
)
from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData

__all__ = ["BaseImageNode"]


# 常见图像模型的粗略显存估算（fp16，GB），用于 describe() 与调度器
_VRAM_ESTIMATES: dict[str, float] = {
    "stabilityai/stable-diffusion-xl-base-1.0": 8.0,
    "stabilityai/stable-diffusion-xl-refiner-1.0": 7.0,
    "stabilityai/stable-diffusion-x4-upscaler": 6.0,
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": 8.0,
    "briaai/RMBG-2.0": 1.0,
    "runwayml/stable-diffusion-v1-5": 4.0,
    "stabilityai/stable-diffusion-2-1": 5.0,
}

# 许可证信息
_LICENSE_INFO: dict[str, str] = {
    "stabilityai/stable-diffusion-xl-base-1.0": "OpenRAIL++-M (CreativeML Open RAIL++-M License)",
    "stabilityai/stable-diffusion-xl-refiner-1.0": "OpenRAIL++-M",
    "stabilityai/stable-diffusion-x4-upscaler": "OpenRAIL++-M",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": "OpenRAIL++-M",
    "briaai/RMBG-2.0": "BRIA RMBG-2.0 Community License",
}


class BaseImageNode(Node):
    """图像域节点抽象基类。

    封装基于 ``diffusers`` 的图像生成 Pipeline 加载与推理流程。子类需实现
    :meth:`run`，并通过类属性声明 ``name``/``description``/
    ``input_types``/``output_types``，同时覆写 :meth:`_load_pipeline` 指定
    具体加载哪个 diffusers Pipeline。

    Parameters
    ----------
    model:
        HuggingFace 模型标识或本地路径。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，可选 ``"float16"``/``"float32"``/``"bfloat16"``，
        默认 ``"float16"``。
    enable_attention_slicing:
        是否启用 attention slicing 以节省显存，默认 ``True``。
    enable_vae_slicing:
        是否启用 VAE slicing 以节省显存，默认 ``True``。
    enable_model_cpu_offload:
        是否启用模型 CPU offload（逐模块迁移到 GPU），默认 ``False``。
        注意：启用 cpu_offload 时 VAE 仍会被上转为 float32（防黑图），
        decode 阶段显存峰值约为 float16 的 2 倍。显存紧张时可关闭此选项
        并改用 dtype="float32" 整体降精度。
    scheduler_name:
        可选的调度器名称（如 ``"EulerDiscreteScheduler"``），``None``
        表示使用模型默认调度器。
    scheduler:
        显存调度器实例，``None`` 使用全局单例。
    bus:
        事件总线实例，``None`` 使用全局单例。
    """

    domain: str = "image"
    description: str = "Base image node."
    version: str = "0.1.0"
    input_types: list[str] = ["image", "mosaic"]
    output_types: list[str] = ["image"]

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        dtype: str = "float16",
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_model_cpu_offload: bool = False,
        scheduler_name: str | None = None,
        scheduler: Scheduler | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bus=bus, **kwargs)
        self._model_name: str = model
        self._scheduler_name: str | None = scheduler_name
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._logger = logging.getLogger(f"mosaic.nodes.image.{self.name}")

        # 自动解析设备与 dtype：CPU 环境下将 float16 降级为 float32
        # （float16 在 CPU 上无法正确推理，会产生黑图）
        self._device, self._dtype_str = auto_resolve_device_dtype(
            device, dtype, self._scheduler, self._logger,
        )
        self._enable_attention_slicing: bool = enable_attention_slicing
        self._enable_vae_slicing: bool = enable_vae_slicing
        self._enable_model_cpu_offload: bool = enable_model_cpu_offload

        # 运行时持有的 diffusers Pipeline（load 后填充）
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载 diffusers Pipeline 到 GPU/CPU。

        通过 ``Scheduler.track`` 注册显存跟踪后执行实际加载。本方法由
        ``Scheduler.ensure_loaded`` 回调，不应在其中调用 ``ensure_loaded``
        以免递归。
        """
        self._scheduler.track(self)

        if self._pipeline is not None:
            self._loaded = True
            return

        self._logger.info("Loading pipeline for model %s ...", self._model_name)
        self._load_pipeline()
        # 上转 VAE + text_encoder（SD 1.5）为 float32，防止 float16 下产生黑图/NaN
        upcast_pipeline_components(self._pipeline, self._model_name, self._logger)
        self._loaded = True

    def _upcast_vae_fp32(self) -> None:
        """[已弃用] 请使用 upcast_pipeline_components()。

        保留仅为向后兼容（子类可能直接调用此方法）。
        """
        upcast_pipeline_components(self._pipeline, self._model_name, self._logger)

    @abc.abstractmethod
    def _load_pipeline(self) -> None:
        """子类实现：实际加载 diffusers Pipeline。

        子类应在此方法中：
        1. 惰性导入 ``diffusers`` 相关 Pipeline 类与 ``torch``；
        2. 调用 ``from_pretrained`` 加载模型；
        3. 应用显存优化（attention_slicing / vae_slicing / cpu_offload）；
        4. 可选切换调度器；
        5. 将 Pipeline 赋值给 ``self._pipeline``。
        """

    def unload(self) -> None:
        """释放 diffusers Pipeline。

        本方法执行实际资源清理。它由 ``Scheduler.release`` /
        ``Scheduler._evict`` 回调，不应在其中调用
        ``scheduler.release(self)`` 以免递归。
        """
        self._pipeline = None
        self._loaded = False
        self._logger.info("Pipeline for model %s unloaded.", self._model_name)

    # ------------------------------------------------------------------
    # 公共推理逻辑
    # ------------------------------------------------------------------
    def _infer_device(self) -> str:
        """推断推理设备。"""
        if self._pipeline is None:
            return self._scheduler.device
        # 优先从 pipeline 的 UNet 参数推断设备，失败时由工具函数回退到调度器设备
        unet = getattr(self._pipeline, "unet", None)
        return infer_device(
            unet if unet is not None else self._pipeline, self._scheduler
        )

    def _resolve_dtype(self) -> Any:
        """解析 torch dtype 字符串为 torch.dtype 对象。"""
        return resolve_dtype(self._dtype_str)

    def _apply_optimizations(self) -> None:
        """对已加载的 Pipeline 应用显存优化配置。

        委托给 :func:`mosaic.core._device_utils.apply_optimizations`，兼容
        diffusers 0.40+ 的 VAE slicing API 变更（优先 ``pipe.vae.enable_slicing()``，
        回退 ``pipe.enable_vae_slicing()``）。
        """
        apply_optimizations(
            self._pipeline,
            enable_cpu_offload=self._enable_model_cpu_offload,
            enable_attention_slicing=self._enable_attention_slicing,
            enable_vae_slicing=self._enable_vae_slicing,
        )

    def _switch_scheduler(self) -> None:
        """可选：切换 Pipeline 的调度器。"""
        if self._scheduler_name is None or self._pipeline is None:
            return
        try:
            from diffusers import scheduler_map  # type: ignore

            sched_cls = scheduler_map.get(self._scheduler_name)
            if sched_cls is None:
                self._logger.warning(
                    "Unknown scheduler %r, keeping default.", self._scheduler_name
                )
                return
            # 从当前配置创建新调度器
            config = self._pipeline.scheduler.config
            self._pipeline.scheduler = sched_cls.from_config(config)
            self._logger.info("Switched scheduler to %s.", self._scheduler_name)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to switch scheduler: %s", exc)

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

        # B2: pipeline 未加载时不应进入推理流程，此处防御性断言
        if self._pipeline is None:
            self._logger.warning(
                "_prepare_seed called before pipeline loaded; using config device."
            )

        # B1: cpu_offload 时 UNet 在 CPU，_infer_device 返回 "cpu"，
        # 但 pipeline 实际执行时会在 GPU。使用配置设备（self._device）更可靠。
        device = self._device
        try:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        except (RuntimeError, ValueError, TypeError):
            # CPU-only 环境可能不支持 cuda generator
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)

        return seed, generator

    def _run_pipeline(self, **kwargs: Any) -> Any:
        """在 torch.inference_mode 下执行 Pipeline 调用。

        Parameters
        ----------
        **kwargs:
            透传给 ``self._pipeline.__call__`` 的参数。

        Returns
        -------
        Any
            Pipeline 输出（通常是 ``StableDiffusionPipelineOutput`` 或类似）。
        """
        return run_diffusers_pipeline(self._pipeline, **kwargs)

    # ------------------------------------------------------------------
    # 图像前后处理工具
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_pil_image(image: Any) -> Any:
        """确保输入是 PIL.Image，如果是 numpy 数组则转换。"""
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            # 防御性 dtype 转换：PIL 需要 uint8
            if image.dtype != np.uint8:
                image = np.clip(
                    image * 255 if image.max() <= 1.0 else image,
                    0, 255,
                ).astype(np.uint8)
            return Image.fromarray(image)
        raise TypeError(
            f"Expected PIL.Image.Image or numpy.ndarray, got {type(image).__name__}."
        )

    @staticmethod
    def _resize_to_multiple_of_8(
        image: Any, target_size: tuple[int, int] | None = None
    ) -> Any:
        """将图像尺寸调整为 8 的倍数（diffusers 要求）。

        Parameters
        ----------
        image:
            PIL.Image 实例。
        target_size:
            目标尺寸 ``(width, height)``。``None`` 表示按原图最近 8 倍数对齐。
        """
        from PIL import Image  # type: ignore

        if not isinstance(image, Image.Image):
            image = BaseImageNode._ensure_pil_image(image)

        if target_size is not None:
            w, h = target_size
        else:
            w, h = image.size

        # 对齐到 8 的倍数
        w = max(8, (w // 8) * 8)
        h = max(8, (h // 8) * 8)

        if (w, h) != image.size:
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _binarize_mask(mask_image: Any, threshold: int = 127) -> Any:
        """将灰度 mask 二值化（白色=待重绘区域）。

        Parameters
        ----------
        mask_image:
            PIL.Image 实例（灰度或 RGB）。
        threshold:
            二值化阈值，像素值 > threshold 视为前景（255），否则背景（0）。
        """
        from PIL import Image  # type: ignore

        if not isinstance(mask_image, Image.Image):
            mask_image = BaseImageNode._ensure_pil_image(mask_image)

        gray = mask_image.convert("L")
        # 应用阈值
        return gray.point(lambda p: 255 if p > threshold else 0)

    @staticmethod
    def _limit_image_size(image: Any, max_side: int = 512) -> Any:
        """限制图像最长边不超过 max_side，按比例缩放。"""
        from PIL import Image  # type: ignore

        if not isinstance(image, Image.Image):
            image = BaseImageNode._ensure_pil_image(image)

        w, h = image.size
        longest = max(w, h)
        if longest <= max_side:
            return image

        ratio = max_side / longest
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        # 对齐到 8 的倍数
        new_w = max(8, (new_w // 8) * 8)
        new_h = max(8, (new_h // 8) * 8)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # ------------------------------------------------------------------
    # Node 抽象方法
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行节点逻辑（子类实现）。"""

    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息。"""
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=self._build_model_info(),
        )

    def _build_model_info(self) -> dict[str, Any]:
        """构造模型信息字典。"""
        vram = _VRAM_ESTIMATES.get(self._model_name, 8.0)
        license_info = _LICENSE_INFO.get(
            self._model_name, "See model card on HuggingFace"
        )
        return {
            "name": self._model_name,
            "source": "HuggingFace",
            "license": license_info,
            "vram_gb": vram,
            "dtype": self._dtype_str,
            "device": self._device,
            "attention_slicing": self._enable_attention_slicing,
            "vae_slicing": self._enable_vae_slicing,
            "cpu_offload": self._enable_model_cpu_offload,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"model={self._model_name!r} state={status}>"
        )
