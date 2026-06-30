# mosaic/nodes/consistency/identity_keeper.py
"""IdentityKeeper 身份保持节点。

根据参考人脸图像生成保持身份特征（五官、面部结构）的新图像。
支持三种身份保持方法：

* ``"instantid"``        —— 基于 InstantID（Identity ControlNet + Face IP-Adapter），
  身份保持能力最强。
* ``"ip-adapter-face"``  —— 基于 IP-Adapter Face，轻量、通用，对基础模型侵入小。
* ``"photomaker"``       —— 基于 PhotoMaker，可将多张参考图叠加为统一身份。

设计要点
--------
* 继承 :class:`BaseConsistencyNode`，复用其图像前后处理、显存优化、事件发射与
  随机种子准备逻辑。
* ``diffusers`` / ``torch`` / ``PIL`` / ``insightface`` 全部惰性导入，使本模块在
  依赖缺失时仍可被注册表发现与导入。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理：``run`` 调用
  ``ensure_loaded`` 触发按需加载 + LRU 淘汰。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出 ``node_start``/
  ``node_complete``/``node_error`` 事件。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.consistency._base import BaseConsistencyNode

__all__ = ["IdentityKeeper"]


# 支持的身份保持方法
_SUPPORTED_METHODS: tuple[str, ...] = (
    "instantid",
    "ip-adapter-face",
    "photomaker",
)

# IP-Adapter Face 权重仓库与文件名（SDXL）
_IP_ADAPTER_FACE_REPO = "h94/IP-Adapter"
_IP_ADAPTER_FACE_SUBFOLDER = "sdxl_models"
_IP_ADAPTER_FACE_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"

# 各 method 推荐的默认基础模型（当用户未显式指定时使用）
_METHOD_DEFAULT_MODELS = {
    "instantid": "InstantX/InstantID",
    "ip-adapter-face": "stabilityai/stable-diffusion-xl-base-1.0",
    "photomaker": "TencentARC/PhotoMaker",
}


@registry.register
class IdentityKeeper(BaseConsistencyNode):
    """身份保持节点：根据参考人脸生成保持身份特征的新图像。

    Parameters
    ----------
    model:
        HuggingFace 模型标识或本地路径。不同 ``method`` 推荐的默认模型：

        * ``"instantid"``       -> ``"InstantX/InstantID"``
        * ``"ip-adapter-face"`` -> ``"stabilityai/stable-diffusion-xl-base-1.0"``
        * ``"photomaker"``      -> ``"TencentARC/PhotoMaker"``
    method:
        身份保持方法，可选 ``"instantid"`` / ``"ip-adapter-face"`` /
        ``"photomaker"``，默认 ``"instantid"``。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，可选 ``"float16"`` / ``"float32"`` / ``"bfloat16"``，
        默认 ``"float16"``。
    **kwargs:
        透传给 :class:`BaseConsistencyNode` 的参数（``scheduler`` / ``bus`` 等）。

    Limitations
    -----------
    * 需要 ``diffusers``、``torch``、``Pillow``；人脸检测优先使用 ``insightface``，
      缺失时回退到中心裁剪（精度下降）。
    * 原生 :class:`~diffusers.StableDiffusionXLInstantIDPipeline` 需要
      ``diffusers >= 0.29``；低版本自动回退到
      ``StableDiffusionXLControlNetPipeline + ControlNet``，身份保持效果略弱。
    * ``identity_score`` 基于结构相似度（SSIM）近似计算，并非真正的人脸识别
      匹配分数，仅用于粗略量化身份一致性。
    * InstantID 方法需要参考图中存在可检测的人脸，否则抛出 :class:`ValueError`。
    * GPU 强烈推荐；CPU 模式下推理极慢。

    Examples
    --------
    >>> keeper = IdentityKeeper(method="instantid")
    >>> result = keeper(MosaicData(
    ...     reference_image="face.jpg",
    ...     prompt="a portrait photo of a person wearing a suit, studio lighting",
    ...     negative_prompt="blurry, low quality, deformed",
    ...     identity_strength=0.8,
    ...     seed=42,
    ... ))
    >>> result["image"].save("portrait.png")
    >>> print(result["identity_score"])
    """

    name: str = "identity-keeper"
    description: str = (
        "Generate identity-preserving images from a reference face using "
        "InstantID, IP-Adapter Face, or PhotoMaker. Computes an approximate "
        "identity consistency score between the reference and the output."
    )
    version: str = "0.1.0"
    input_types: list[str] = ["image", "mosaic"]
    output_types: list[str] = ["image"]

    def __init__(
        self,
        model: str = "InstantX/InstantID",
        method: str = "instantid",
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method {method!r}. "
                f"Choose from {_SUPPORTED_METHODS}."
            )
        super().__init__(device=device, dtype=dtype, **kwargs)
        self._method: str = method
        # 若用户使用了某方法的默认模型标识之外的值，仍以用户值为准；
        # 否则采用对应 method 的推荐模型。
        self._model_name: str = (
            model if model != "InstantX/InstantID" or method == "instantid"
            else _METHOD_DEFAULT_MODELS[method]
        )
        # 运行时标记：instantid 是否加载了原生 Pipeline
        self._native_instantid: bool = False

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载身份保持 Pipeline。

        通过 :meth:`Scheduler.track` 注册显存跟踪后，根据 ``method`` 分发到
        具体的加载器。本方法由 ``Scheduler.ensure_loaded`` 回调，不应在其中
        调用 ``ensure_loaded`` 以免递归。
        """
        self._scheduler.track(self)

        if self._pipeline is not None:
            self._loaded = True
            return

        self._logger.info(
            "Loading IdentityKeeper pipeline (method=%s, model=%s) ...",
            self._method,
            self._model_name,
        )

        if self._method == "instantid":
            self._load_instantid()
        elif self._method == "ip-adapter-face":
            self._load_ip_adapter_face()
        else:  # photomaker
            self._load_photomaker()

        self._apply_optimizations()
        self._loaded = True

    def _resolve_dtype_and_variant(self) -> tuple[Any, str | None]:
        """返回 (torch.dtype, variant) 用于 from_pretrained。"""
        torch_dtype = self._resolve_dtype()
        variant = "fp16" if self._dtype_str in ("float16", "fp16") else None
        return torch_dtype, variant

    def _move_pipeline_to_device(self) -> None:
        """将 Pipeline 迁移到目标设备，CUDA 不可用时回退到 CPU。"""
        target = self._resolve_device()
        try:
            self._pipeline = self._pipeline.to(target)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to move pipeline to %s: %s. Falling back to CPU.",
                target,
                exc,
            )
            self._device = "cpu"
            try:
                self._pipeline = self._pipeline.to("cpu")
            except Exception:  # noqa: BLE001
                pass

    def _load_instantid(self) -> None:
        """加载 InstantID Pipeline。

        优先使用 ``StableDiffusionXLInstantIDPipeline``；若 diffusers 版本不含
        该类则回退到 ``StableDiffusionXLControlNetPipeline + ControlNetModel``。
        """
        torch_dtype, variant = self._resolve_dtype_and_variant()

        try:
            from diffusers import StableDiffusionXLInstantIDPipeline  # type: ignore

            self._pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                variant=variant,
            )
            self._native_instantid = True
        except (ImportError, AttributeError, ValueError, OSError):
            self._logger.warning(
                "StableDiffusionXLInstantIDPipeline not available "
                "(diffusers too old or model incompatible). "
                "Falling back to StableDiffusionXLControlNetPipeline + ControlNet."
            )
            from diffusers import (  # type: ignore
                ControlNetModel,
                StableDiffusionXLControlNetPipeline,
            )

            controlnet = ControlNetModel.from_pretrained(
                self._model_name,
                subfolder="ControlNetModel",
                torch_dtype=torch_dtype,
                variant=variant,
            )
            self._pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                self._model_name,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                variant=variant,
            )
            self._native_instantid = False

        self._move_pipeline_to_device()
        self._logger.info(
            "InstantID pipeline loaded (native=%s, dtype=%s, device=%s).",
            self._native_instantid,
            self._dtype_str,
            self._device,
        )

    def _load_ip_adapter_face(self) -> None:
        """加载 SDXL Pipeline 并挂载 IP-Adapter Face 权重。"""
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        torch_dtype, variant = self._resolve_dtype_and_variant()

        from mosaic.nodes._pipeline_utils import _build_error_message

        try:
            self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                variant=variant,
            )
        except (
            ImportError,
            AttributeError,
            ValueError,
            OSError,
            EnvironmentError,
        ) as exc:
            raise RuntimeError(
                _build_error_message(self._model_name, exc)
            ) from exc
        self._move_pipeline_to_device()

        # 加载 IP-Adapter Face 权重
        try:
            self._pipeline.load_ip_adapter(
                _IP_ADAPTER_FACE_REPO,
                subfolder=_IP_ADAPTER_FACE_SUBFOLDER,
                weight_name=_IP_ADAPTER_FACE_WEIGHT,
            )
            self._logger.info(
                "IP-Adapter Face weights loaded from %s/%s.",
                _IP_ADAPTER_FACE_REPO,
                _IP_ADAPTER_FACE_WEIGHT,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load IP-Adapter Face weights: %s. "
                "Identity preservation will be disabled.", exc,
            )

    def _load_photomaker(self) -> None:
        """加载 PhotoMaker Pipeline。"""
        from diffusers import PhotoMakerPipeline  # type: ignore

        torch_dtype, variant = self._resolve_dtype_and_variant()

        from mosaic.nodes._pipeline_utils import _build_error_message

        try:
            self._pipeline = PhotoMakerPipeline.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                variant=variant,
            )
        except (
            ImportError,
            AttributeError,
            ValueError,
            OSError,
            EnvironmentError,
        ) as exc:
            raise RuntimeError(
                _build_error_message(self._model_name, exc)
            ) from exc
        self._move_pipeline_to_device()
        self._logger.info(
            "PhotoMaker pipeline loaded (dtype=%s, device=%s).",
            self._dtype_str,
            self._device,
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行身份保持生成。

        Parameters
        ----------
        input_data:
            必须包含 ``reference_image`` (``PIL.Image`` 或文件路径 str) 与
            ``prompt`` (str)；可选 ``negative_prompt`` (str)、``width``
            (int, 默认 1024)、``height`` (int, 默认 1024)、
            ``num_inference_steps`` (int, 默认 30)、``guidance_scale``
            (float, 默认 5.0)、``identity_strength`` (float, 默认 0.8)、
            ``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``image`` (PIL.Image)、``reference_image`` (PIL.Image)、
            ``identity_score`` (float)、``seed`` (int)、``method`` (str)、
            ``model_name`` (str)。

        Raises
        ------
        ValueError
            缺少 ``reference_image`` / ``prompt``，或参考图中检测不到人脸。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # ---------- 校验输入 ----------
            reference_image = input_data.get("reference_image")
            if reference_image is None:
                raise ValueError(
                    "IdentityKeeper requires 'reference_image' "
                    "(PIL.Image or file path)."
                )
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(
                    f"IdentityKeeper requires 'prompt' (str), "
                    f"got {type(prompt).__name__}."
                )

            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            # ---------- 提取参数 ----------
            width = int(input_data.get("width", 1024))
            height = int(input_data.get("height", 1024))
            width = max(8, (width // 8) * 8)
            height = max(8, (height // 8) * 8)
            # 大图像内存保护
            self._check_image_dimensions(width, height)

            num_inference_steps = int(input_data.get("num_inference_steps", 30))
            guidance_scale = float(input_data.get("guidance_scale", 5.0))
            identity_strength = float(input_data.get("identity_strength", 0.8))
            identity_strength = max(0.0, min(1.0, identity_strength))

            # ---------- 图像前处理 ----------
            ref_img = self._load_image(reference_image)
            # 提取人脸区域；检测不到人脸时 _prepare_face_region 抛 ValueError
            face_image, _bbox = self._prepare_face_region(ref_img)

            # ---------- 种子 ----------
            seed, generator = self._prepare_seed(input_data.get("seed"))

            # ---------- 分发推理 ----------
            self._emit_progress(
                current=0, total=1, message="Generating identity-preserving image"
            )
            if self._method == "instantid":
                image = self._generate_instantid(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ref_img=ref_img,
                    face_image=face_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    identity_strength=identity_strength,
                    generator=generator,
                )
            elif self._method == "ip-adapter-face":
                image = self._generate_ip_adapter_face(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ref_img=ref_img,
                    face_image=face_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    identity_strength=identity_strength,
                    generator=generator,
                )
            else:  # photomaker
                image = self._generate_photomaker(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ref_img=ref_img,
                    face_image=face_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    identity_strength=identity_strength,
                    generator=generator,
                )
            self._emit_progress(
                current=1, total=1, message="Generation complete"
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        # ---------- 身份一致性分数 ----------
        identity_score = self._compute_image_similarity(image, ref_img)

        result = MosaicData(
            image=image,
            reference_image=ref_img,
            identity_score=identity_score,
            seed=seed,
            method=self._method,
            model_name=self._model_name,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "identity_score": identity_score,
                "seed": seed,
                "method": self._method,
                "width": width,
                "height": height,
            },
        )
        return result

    def _generate_instantid(
        self,
        prompt: str,
        negative_prompt: str | None,
        ref_img: Any,
        face_image: Any,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        identity_strength: float,
        generator: Any,
    ) -> Any:
        """InstantID 方法推理。"""
        pipe_kwargs: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            # IP-Adapter 输入：用于提取人脸嵌入
            "ip_adapter_image": ref_img,
        }
        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = negative_prompt

        if self._native_instantid:
            # 原生 InstantID Pipeline
            pipe_kwargs["image"] = face_image
            pipe_kwargs["identitynet_strength"] = identity_strength
            pipe_kwargs["adapter_strength_ratio"] = identity_strength
        else:
            # 回退：SDXL + ControlNet
            pipe_kwargs["image"] = face_image
            pipe_kwargs["controlnet_conditioning_scale"] = identity_strength

        output = self._run_pipeline(**pipe_kwargs)
        return self._extract_image(output)

    def _generate_ip_adapter_face(
        self,
        prompt: str,
        negative_prompt: str | None,
        ref_img: Any,
        face_image: Any,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        identity_strength: float,
        generator: Any,
    ) -> Any:
        """IP-Adapter Face 方法推理。"""
        # 设置 IP-Adapter 强度
        self._set_ip_adapter_scale(identity_strength)

        pipe_kwargs: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "ip_adapter_image": ref_img,
        }
        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = negative_prompt

        output = self._run_pipeline(**pipe_kwargs)
        return self._extract_image(output)

    def _generate_photomaker(
        self,
        prompt: str,
        negative_prompt: str | None,
        ref_img: Any,
        face_image: Any,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        identity_strength: float,
        generator: Any,
    ) -> Any:
        """PhotoMaker 方法推理。"""
        # PhotoMaker 需要触发词 "img" 作为身份占位符
        pm_prompt = prompt
        if "img" not in pm_prompt.lower():
            pm_prompt = f"img, {pm_prompt}"

        # start_merge_step 越小，身份注入越早、越强
        start_merge_step = max(
            1, int(num_inference_steps * (1.0 - identity_strength))
        )

        pipe_kwargs: dict = {
            "prompt": pm_prompt,
            "input_id_images": [ref_img],
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "pm_alpha_scale": identity_strength,
            "start_merge_step": start_merge_step,
        }
        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = negative_prompt

        output = self._run_pipeline(**pipe_kwargs)
        return self._extract_image(output)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _set_ip_adapter_scale(self, scale: float) -> None:
        """设置 IP-Adapter 强度，兼容标量与列表两种 API。"""
        if self._pipeline is None or not hasattr(
            self._pipeline, "set_ip_adapter_scale"
        ):
            return
        try:
            self._pipeline.set_ip_adapter_scale(scale)
        except (TypeError, ValueError):
            try:
                self._pipeline.set_ip_adapter_scale([scale])
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to set ip_adapter_scale=%s: %s", scale, exc
                )

    @staticmethod
    def _extract_image(output: Any) -> Any:
        """从 Pipeline 输出中提取首张图片。"""
        images = getattr(output, "images", None)
        if images:
            return images[0]
        # 兼容返回 dict 的 Pipeline
        if isinstance(output, dict) and output.get("images"):
            return output["images"][0]
        raise RuntimeError(
            "Pipeline output did not contain any image."
        )

    # ------------------------------------------------------------------
    # 卸载 / 规格
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """释放 Pipeline 资源。"""
        self._pipeline = None
        self._native_instantid = False
        self._loaded = False
        self._logger.info(
            "IdentityKeeper pipeline unloaded (model=%s).", self._model_name
        )

    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息（VRAM、许可证、方法）。"""
        model_info = self._build_model_info(self._model_name)
        model_info["method"] = self._method
        model_info["native_instantid"] = self._native_instantid
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=model_info,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<IdentityKeeper name={self.name!r} method={self._method!r} "
            f"model={self._model_name!r} state={status}>"
        )
