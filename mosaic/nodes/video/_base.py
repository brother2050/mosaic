# mosaic/nodes/video/_base.py
"""视频域节点基类。

提取视频生成 / 处理节点共用的模型加载、推理与视频前后处理逻辑。
子类只需实现 :meth:`BaseVideoNode.run` 与 :meth:`_load_model`，底层
推理流程与帧处理工具由本基类提供。

设计要点
--------
* ``diffusers`` / ``torch`` / ``imageio`` 均采用惰性导入，使本模块在未
  安装这些依赖时仍可被注册表发现与导入（仅在实际加载 / 推理时才报
  依赖缺失）。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理：
  ``load`` 调用 ``scheduler.track(self)`` 注册显存跟踪并执行实际加载；
  ``run`` 调用 ``scheduler.ensure_loaded(self)`` 触发按需加载 + LRU 淘汰。
  注意：``load`` 不能调用 ``ensure_loaded``（会递归）。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出
  ``node_start`` / ``node_complete`` / ``node_error`` / ``progress`` 事件。
* 提供统一的视频前后处理工具：加载、保存、抽帧、resize、帧-tensor
  转换、按时间戳取帧、偶数尺寸对齐。
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any

from mosaic.core._device_utils import (
    auto_resolve_device_dtype,
    infer_device,
    resolve_device,
    resolve_dtype,
    upcast_pipeline_components,
)
from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData, VideoData

__all__ = ["BaseVideoNode"]


# 常见视频模型的粗略显存估算（fp16，GB），用于 describe() 与调度器
_VRAM_ESTIMATES: dict[str, float] = {
    "THUDM/CogVideoX-5b": 18.0,
    "THUDM/CogVideoX-2b": 9.0,
    "stabilityai/stable-video-diffusion-img2vid": 10.0,
    "stabilityai/stable-video-diffusion-img2vid-xt": 12.0,
    "ali-vilab/i2vgen-xl": 16.0,
    # Wan2.1 系列
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": 30.0,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": 8.0,
    "Wan-AI/Wan2.1-T2V-14B": 30.0,
    "Wan-AI/Wan2.1-T2V-1.3B": 8.0,
    # Wan2.2 系列
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": 30.0,
    "Wan-AI/Wan2.2-T2V-A14B": 30.0,
    # HunyuanVideo
    "hunyuanvideo-community/HunyuanVideo": 60.0,
    # LTX-Video
    "Lightricks/LTX-Video": 12.0,
    "Lightricks/LTX-Video-13B": 30.0,
}

# 许可证信息
_LICENSE_INFO: dict[str, str] = {
    "THUDM/CogVideoX-5b": "CogVideoX License (Apache 2.0)",
    "THUDM/CogVideoX-2b": "CogVideoX License (Apache 2.0)",
    "stabilityai/stable-video-diffusion-img2vid": "Stability AI Community License",
    "stabilityai/stable-video-diffusion-img2vid-xt": "Stability AI Community License",
    "ali-vilab/i2vgen-xl": "Tongyi Lab License",
    # Wan2.1 / Wan2.2 系列：Apache 2.0
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "Apache 2.0",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "Apache 2.0",
    "Wan-AI/Wan2.1-T2V-14B": "Apache 2.0",
    "Wan-AI/Wan2.1-T2V-1.3B": "Apache 2.0",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "Apache 2.0",
    "Wan-AI/Wan2.2-T2V-A14B": "Apache 2.0",
    # HunyuanVideo：Tencent License
    "hunyuanvideo-community/HunyuanVideo": "Tencent Hunyuan Video License",
    # LTX-Video：OpenRAIL-M
    "Lightricks/LTX-Video": "OpenRAIL-M License",
    "Lightricks/LTX-Video-13B": "OpenRAIL-M License",
}


class BaseVideoNode(Node):
    """视频域节点抽象基类。

    封装基于 ``diffusers`` 的视频生成 Pipeline 加载与推理流程，以及通用
    的视频前后处理工具。子类需实现 :meth:`run` 与 :meth:`_load_model`。

    Parameters
    ----------
    model:
        HuggingFace 模型标识或本地路径。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，可选 ``"float16"`` / ``"float32"`` / ``"bfloat16"``，
        默认 ``"float16"``。
    scheduler:
        显存调度器实例，``None`` 使用全局单例。
    bus:
        事件总线实例，``None`` 使用全局单例。
    """

    domain: str = "video"
    description: str = "Base video node."
    version: str = "0.1.0"
    input_types: list[str] = ["text", "image", "video", "mosaic"]
    output_types: list[str] = ["video"]

    def __init__(
        self,
        model: str = "",
        device: str = "cuda",
        dtype: str = "float16",
        scheduler: Scheduler | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bus=bus, **kwargs)
        self._model_name: str = model
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._logger = logging.getLogger(f"mosaic.nodes.video.{self.name}")

        # 自动解析设备与 dtype：CPU/SD1.5 环境下将 float16 降级为 float32
        self._device, self._dtype_str = auto_resolve_device_dtype(
            device, dtype, self._scheduler, self._logger,
            model_name=model,
        )

        # 运行时持有的 Pipeline / 模型（load 后填充）
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载视频模型到 GPU/CPU。

        通过 ``Scheduler.track`` 注册显存跟踪后执行实际加载。本方法由
        ``Scheduler.ensure_loaded`` 回调，不应在其中调用 ``ensure_loaded``
        以免递归。
        """
        self._scheduler.track(self)

        if self._pipeline is not None:
            self._loaded = True
            return

        self._logger.info("Loading video model %s ...", self._model_name)
        self._load_model()
        # 上转 VAE 为 float32，防止 float16 下产生黑图/NaN
        upcast_pipeline_components(self._pipeline, self._model_name, self._logger)
        self._loaded = True

    def _upcast_vae_fp32(self) -> None:
        """[已弃用] 请使用 upcast_pipeline_components()。"""
        upcast_pipeline_components(self._pipeline, self._model_name, self._logger)

    @abc.abstractmethod
    def _load_model(self) -> None:
        """子类实现：实际加载模型。

        子类应在此方法中：
        1. 惰性导入所需的库（diffusers / torch / imageio 等）；
        2. 加载 Pipeline / 模型；
        3. 迁移到目标设备；
        4. 将 Pipeline 赋值给 ``self._pipeline``。
        """

    def unload(self) -> None:
        """释放视频模型。

        本方法执行实际资源清理。它由 ``Scheduler.release`` /
        ``Scheduler._evict`` 回调，不应在其中调用
        ``scheduler.release(self)`` 以免递归。
        """
        self._pipeline = None
        self._loaded = False
        self._logger.info("Video model %s unloaded.", self._model_name)

    # ------------------------------------------------------------------
    # 设备与推理辅助
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

    def _resolve_device(self) -> str:
        """解析实际设备字符串，无 GPU 时降级到 CPU。"""
        device = resolve_device(self._device)
        if device != self._device:
            self._logger.warning(
                "CUDA not available, falling back to CPU for %s.",
                self.name,
            )
        return device

    def _resolve_dtype(self) -> Any:
        """解析 torch dtype 字符串为 torch.dtype 对象。"""
        return resolve_dtype(self._dtype_str)

    # ------------------------------------------------------------------
    # 视频前后处理工具
    # ------------------------------------------------------------------
    @staticmethod
    def _load_video(path: str) -> VideoData:
        """从文件加载视频，逐帧读取为 PIL.Image 列表。

        使用 ``imageio`` 读取视频文件。

        Parameters
        ----------
        path:
            视频文件路径（mp4 / avi / mov 等）。

        Returns
        -------
        VideoData
            包含 ``frames`` / ``fps`` / ``metadata`` 的视频数据。

        Raises
        ------
        ImportError
            未安装 ``imageio`` 时抛出。
        FileNotFoundError
            文件不存在时抛出。
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        import imageio.v2 as imageio  # type: ignore
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        fps = int(meta.get("fps", 30))

        frames: list[Any] = []
        for frame in reader:
            # imageio 返回 numpy 数组 (H, W, C)
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)

        reader.close()

        width, height = frames[0].size if frames else (0, 0)
        duration = len(frames) / fps if fps > 0 else 0.0

        return VideoData(
            frames=frames,
            fps=fps,
            metadata={
                "source": path,
                "width": width,
                "height": height,
                "duration": duration,
                "frame_count": len(frames),
            },
        )

    @staticmethod
    def _save_video(
        frames: list[Any],
        fps: int,
        path: str,
        codec: str = "libx264",
    ) -> None:
        """保存帧列表为视频文件。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 列表。
        fps:
            帧率。
        path:
            输出文件路径。
        codec:
            视频编码器，默认 ``"libx264"``。

        Raises
        ------
        ImportError
            未安装 ``imageio`` 时抛出。
        """
        import imageio.v2 as imageio  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        writer = imageio.get_writer(
            path,
            fps=fps,
            codec=codec,
        )

        for frame in frames:
            if isinstance(frame, Image.Image):
                arr = np.array(frame.convert("RGB"))
            else:
                arr = np.asarray(frame)
                # 防御性 dtype 转换：imageio 期望 uint8
                if arr.dtype != np.uint8:
                    # 空数组保护：arr.size == 0 时 arr.max() 会抛 ValueError
                    if arr.size > 0 and arr.max() <= 1.0:
                        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
            writer.append_data(arr)

        writer.close()

    @staticmethod
    def _extract_frames(
        video_data: VideoData,
        indices: list[int],
    ) -> list[Any]:
        """从 VideoData 中提取指定索引的帧。

        Parameters
        ----------
        video_data:
            视频数据。
        indices:
            帧索引列表。

        Returns
        -------
        list[PIL.Image]
            提取的帧列表。
        """
        frames = video_data.frames
        return [frames[i] for i in indices if 0 <= i < len(frames)]

    @staticmethod
    def _resize_frames(
        frames: list[Any],
        target_size: tuple[int, int],
    ) -> list[Any]:
        """批量 resize 帧列表。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 列表。
        target_size:
            目标尺寸 ``(width, height)``。

        Returns
        -------
        list[PIL.Image]
            resize 后的帧列表。
        """
        from PIL import Image  # type: ignore

        return [f.resize(target_size, Image.Resampling.LANCZOS) for f in frames]

    @staticmethod
    def _frames_to_tensor(frames: list[Any]) -> Any:
        """将 PIL.Image 帧列表转为 torch.Tensor。

        输出形状为 ``(N, C, H, W)``，值域 ``[0, 1]``。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 列表。

        Returns
        -------
        torch.Tensor
            形状 ``(N, C, H, W)`` 的张量。
        """
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        arrays = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                arr = np.array(frame.convert("RGB"))
            else:
                arr = np.asarray(frame)
            # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
            arrays.append(arr)

        # (N, C, H, W)
        stacked = np.stack(arrays, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(stacked)

    @staticmethod
    def _tensor_to_frames(tensor: Any) -> list[Any]:
        """将 torch.Tensor 转为 PIL.Image 帧列表。

        输入形状应为 ``(N, C, H, W)`` 或 ``(C, H, W)``。

        Parameters
        ----------
        tensor:
            ``torch.Tensor``，值域 ``[0, 1]``。

        Returns
        -------
        list[PIL.Image]
            帧列表。
        """
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        # 确保在 CPU 上并转为 numpy
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = np.asarray(tensor)

        # 显式转 float32 避免 float16 精度损失（仅对真实 ndarray）
        if isinstance(arr, np.ndarray) and arr.dtype == np.float16:
            arr = arr.astype(np.float32)

        # 空数组保护：arr.size == 0 时 arr.max() 会抛 ValueError
        if arr.size == 0:
            return []

        # 检测 NaN：若上游 VAE 产生 NaN，此处报错而非静默输出黑帧
        if np.isnan(arr).any():
            raise RuntimeError(
                "NaN detected in video output tensor — likely VAE decode failure. "
                "Consider upcasting VAE to float32 or reducing resolution."
            )

        # 归一化到 [0, 255]
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

        frames: list[Any] = []
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]

        for i in range(arr.shape[0]):
            # (C, H, W) -> (H, W, C)
            frame_arr = np.transpose(arr[i], (1, 2, 0))
            frames.append(Image.fromarray(frame_arr))

        return frames

    @staticmethod
    def _get_frame_at(video_data: VideoData, timestamp: float) -> Any:
        """按时间戳取帧。

        Parameters
        ----------
        video_data:
            视频数据。
        timestamp:
            时间戳（秒）。

        Returns
        -------
        PIL.Image
            对应时间戳的帧。
        """
        frames = video_data.frames
        fps = video_data.fps
        if not frames or fps <= 0:
            from PIL import Image  # type: ignore

            return Image.new("RGB", (1, 1))

        idx = int(timestamp * fps)
        idx = max(0, min(idx, len(frames) - 1))
        return frames[idx]

    @staticmethod
    def _ensure_even_dimensions(width: int, height: int) -> tuple[int, int]:
        """确保宽高为偶数（视频编码要求）。

        大多数视频编码器（H.264 / H.265）要求宽高为偶数，否则编码失败。
        本方法将奇数维度减 1 使其为偶数。

        Parameters
        ----------
        width:
            原始宽度。
        height:
            原始高度。

        Returns
        -------
        tuple[int, int]
            调整后的 ``(width, height)``，均为偶数。
        """
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        return max(2, width), max(2, height)

    def _ensure_video_data(
        self,
        frames: list[Any],
        fps: int,
        **extra: Any,
    ) -> VideoData:
        """将帧列表包装为 VideoData。

        Parameters
        ----------
        frames:
            ``PIL.Image`` 帧列表。
        fps:
            帧率。
        **extra:
            额外 metadata 字段。

        Returns
        -------
        VideoData
            包装后的视频数据。
        """
        from PIL import Image  # type: ignore

        width, height = (0, 0)
        if frames:
            width, height = frames[0].size

        duration = len(frames) / fps if fps > 0 else 0.0
        metadata: dict[str, Any] = {
            "duration": duration,
            "width": width,
            "height": height,
            "frame_count": len(frames),
        }
        metadata.update(extra)

        return VideoData(frames=frames, fps=fps, metadata=metadata)

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
        vram = _VRAM_ESTIMATES.get(self._model_name, 10.0)
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
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"model={self._model_name!r} state={status}>"
        )
