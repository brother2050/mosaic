# mosaic/nodes/image/background_remover.py
"""BackgroundRemover 节点 —— 去除图片背景，返回透明背景的主体。

支持两种后端：
1. 基于 ``transformers`` 的图像分割模型（如 ``briaai/RMBG-2.0``），
   通过 ``pipeline("image-segmentation")`` 或直接推理。
2. 基于 ``rembg`` 库的轻量去背景（构造函数传入 ``use_rembg=True``）。

输出自动转为 RGBA 模式（透明背景），同时返回前景遮罩（灰度图）。
去背景模型体量小，不需要 GPU 也能运行。
"""

from __future__ import annotations

import io
import time
from typing import Any

from mosaic.core.events import EventBus, get_event_bus
from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData

from mosaic.nodes.image._base import BaseImageNode, _VRAM_ESTIMATES, _LICENSE_INFO

__all__ = ["BackgroundRemover"]


@registry.register
class BackgroundRemover(BaseImageNode):
    """去背景节点。

    去除图片背景，返回透明背景的主体图像和前景遮罩。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"briaai/RMBG-2.0"``。
    use_rembg:
        是否使用 ``rembg`` 库代替模型推理，默认 ``False``。
        ``rembg`` 是一个轻量的去背景库，无需下载大型模型。
    **kwargs:
        透传给 :class:`BaseImageNode` 的参数。

    Examples
    --------
    >>> from PIL import Image
    >>> remover = BackgroundRemover()
    >>> img = Image.open("photo.jpg")
    >>> result = remover(MosaicData(image=img))
    >>> result["image"].save("transparent.png")  # RGBA 透明背景
    >>> result["mask"].save("mask.png")          # 灰度遮罩
    """

    name: str = "background-remover"
    description: str = (
        "Remove the background from an image, returning a transparent RGBA "
        "image and a foreground mask. Supports both model-based and rembg backends."
    )
    version: str = "0.1.0"
    input_types = ["image", "mosaic"]
    output_types = ["image"]

    # 分割模型期望的输入尺寸（F2：避免魔法数字）
    DEFAULT_INPUT_SIZE: tuple[int, int] = (1024, 1024)

    def __init__(
        self,
        model: str = "briaai/RMBG-2.0",
        use_rembg: bool = False,
        **kwargs: Any,
    ) -> None:
        self._use_rembg: bool = use_rembg
        # rembg 模式不需要大型模型
        if use_rembg:
            kwargs.setdefault("device", "cpu")
            kwargs.setdefault("dtype", "float32")
            kwargs.setdefault("enable_attention_slicing", False)
            kwargs.setdefault("enable_vae_slicing", False)
        super().__init__(model=model, **kwargs)
        # rembg 会话（load 后填充）
        self._rembg_session: Any = None

    def _load_pipeline(self) -> None:
        """加载去背景模型或 rembg 会话。"""
        if self._use_rembg:
            self._load_rembg()
        else:
            self._load_segmentation_model()

    def _load_rembg(self) -> None:
        """使用 rembg 库创建会话。"""
        try:
            from rembg import new_session  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rembg is not installed. Install it via "
                "`pip install rembg` or use use_rembg=False."
            ) from exc

        self._rembg_session = new_session("u2net")
        self._logger.info("rembg session created (model=u2net).")

    def _load_segmentation_model(self) -> None:
        """使用 transformers 加载图像分割模型。"""
        from transformers import AutoModelForImageSegmentation  # type: ignore
        import torch  # type: ignore

        torch_dtype = self._resolve_dtype()

        self._pipeline = AutoModelForImageSegmentation.from_pretrained(
            self._model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self._pipeline = self._pipeline.to(self._device)
        self._pipeline.eval()

        self._logger.info(
            "Segmentation model %s loaded (dtype=%s, device=%s).",
            self._model_name,
            self._dtype_str,
            self._device,
        )

    def run(self, input_data: MosaicData) -> MosaicData:
        """执行去背景。

        Parameters
        ----------
        input_data:
            必须包含 ``image`` (PIL.Image)。

        Returns
        -------
        MosaicData
            包含 ``image`` (PIL.Image, RGBA 模式，透明背景)、
            ``mask`` (PIL.Image, 灰度遮罩)。

        Raises
        ------
        ValueError
            缺少 ``image``。
        TypeError
            ``image`` 不是 PIL.Image。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # 校验输入
            image = input_data.get("image")
            if image is None:
                raise ValueError(
                    f"BackgroundRemover requires 'image' (PIL.Image), "
                    f"got {type(image).__name__}."
                )
            image = self._ensure_pil_image(image)

            # 根据后端执行去背景
            if self._use_rembg:
                result_image, mask = self._remove_bg_rembg(image)
            else:
                result_image, mask = self._remove_bg_model(image)
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        result = MosaicData(
            image=result_image,
            mask=mask,
            model_name=self._model_name if not self._use_rembg else "rembg/u2net",
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "image_size": result_image.size,
                "mode": result_image.mode,
                "backend": "rembg" if self._use_rembg else "model",
            },
        )
        return result

    def _remove_bg_rembg(self, image: Any) -> tuple:
        """使用 rembg 去除背景。"""
        from rembg import remove  # type: ignore
        from PIL import Image  # type: ignore

        # rembg remove 返回 PNG 字节流（含 alpha 通道）
        output_bytes = remove(image, session=self._rembg_session)
        if hasattr(output_bytes, "read"):
            # 已是文件类对象（含 read 方法）
            result_image = Image.open(output_bytes)
        elif isinstance(output_bytes, bytes):
            # 原始字节流需要 io.BytesIO 包装后才能被 PIL 打开
            result_image = Image.open(io.BytesIO(output_bytes))
        else:
            result_image = Image.open(output_bytes)
        result_image = result_image.convert("RGBA")

        # 提取 alpha 通道作为 mask
        r, g, b, alpha = result_image.split()
        mask = alpha.convert("L")

        return result_image, mask

    def _remove_bg_model(self, image: Any) -> tuple:
        """使用 transformers 分割模型去除背景。"""
        import torch  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        from torchvision import transforms  # type: ignore

        # 预处理：resize + normalize
        orig_size = image.size
        # 模型期望的输入尺寸（通常是 1024x1024 或 512x512）
        input_size = self.DEFAULT_INPUT_SIZE

        preprocess = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            # 显式转 float32，避免 float16 tensor 传入 ToPILImage
            preds = self._pipeline(input_tensor)[-1].sigmoid().float().cpu()

        # 后处理：将预测结果 resize 回原图尺寸
        pred = preds[0].squeeze()
        pred_mask = transforms.ToPILImage()(pred)
        pred_mask = pred_mask.resize(orig_size, Image.Resampling.LANCZOS)

        # 将 mask 转换为 numpy 数组
        mask_array = np.array(pred_mask)

        # 创建 RGBA 图像：原图 + mask 作为 alpha 通道
        if image.mode != "RGBA":
            rgba = image.convert("RGBA").copy()
        else:
            rgba = image.copy()

        # 应用 mask 到 alpha 通道
        alpha = Image.fromarray(mask_array, mode="L")
        rgba.putalpha(alpha)

        return rgba, pred_mask

    def _build_model_info(self) -> dict:
        """构造模型信息字典。"""
        if self._use_rembg:
            return {
                "name": "rembg/u2net",
                "source": "rembg",
                "license": "MIT (rembg) / Apache-2.0 (U^2-Net)",
                "vram_gb": 0.5,
                "dtype": "float32",
                "device": self._device,
                "backend": "rembg",
            }
        vram = _VRAM_ESTIMATES.get(self._model_name, 1.0)
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
            "backend": "transformers",
        }
