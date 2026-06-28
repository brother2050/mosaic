# mosaic/nodes/image/__init__.py
"""图像域节点。

导出该域所有节点类。当前包含 6 个节点：

* :class:`TextToImage`       —— 文生图（SDXL）
* :class:`ImageToImage`      —— 图生图 / 风格转换（SDXL Refiner）
* :class:`Inpainting`        —— 局部重绘（SDXL Inpainting）
* :class:`Upscaler`          —— 超分辨率放大（SD x4 Upscaler）
* :class:`BackgroundRemover` —— 去背景（RMBG-2.0 / rembg）
* :class:`Stylizer`          —— 艺术风格化（SDXL Img2Img + IP-Adapter）
"""

from mosaic.nodes.image._base import BaseImageNode
from mosaic.nodes.image.background_remover import BackgroundRemover
from mosaic.nodes.image.image_to_image import ImageToImage
from mosaic.nodes.image.inpainting import Inpainting
from mosaic.nodes.image.stylizer import Stylizer
from mosaic.nodes.image.text_to_image import TextToImage
from mosaic.nodes.image.upscaler import Upscaler

__all__ = [
    "BaseImageNode",
    "TextToImage",
    "ImageToImage",
    "Inpainting",
    "Upscaler",
    "BackgroundRemover",
    "Stylizer",
]
