# mosaic/nodes/consistency/cross_frame_consistency.py
"""CrossFrameConsistency 节点 —— 跨帧角色 / 主体一致性生成。

在多帧（多 prompt）图像生成场景下，保持同一角色或主体在不同帧之间的
视觉一致性。例如生成分镜脚本、绘本插画、连环画等。

支持三种方法（``method``）：

* ``"consistory"`` —— 加载 SDXL Pipeline，通过对 UNet **交叉注意力**
  的 K/V 投影进行缓存与跨帧融合，实现 Consistory 风格的主体一致性。
* ``"story-diffusion"`` —— 加载 SD Pipeline（默认 SD 1.5），对 **自
  注意力** 的 K/V 进行跨帧共享，实现 Story-Diffusion 风格的一致性。
* ``"all-in-one"`` —— 加载 SDXL Pipeline + IP-Adapter，以参考帧或首
  帧作为图像条件注入后续帧，作为跨帧一致的简化方案。

所有方法均通过 :meth:`BaseConsistencyNode._apply_optimizations` 应用
显存优化（attention slicing / VAE slicing），并通过事件总线报告进度。

设计要点
--------
* ``diffusers`` / ``torch`` / ``PIL`` / ``numpy`` 全部惰性导入，使本模
  块在依赖缺失时仍可被注册表发现与导入。
* ``consistency_strength`` 控制一致性约束的强度（0.0=无约束，1.0=强
  烈复用锚帧特征）。
* 一致性分数基于 :meth:`BaseConsistencyNode._compute_image_similarity`
  （结构相似度）计算每帧与锚帧（参考图或首帧）的相似度。

Limitations
-----------
跨帧一致性是**前沿研究领域**，本节点实现的是上述方法的**简化版本**：

1. KV 共享仅在 UNet 注意力模块的 K/V 线性投影层面进行融合，且锚帧的
   KV 取自去噪的最后一步并用于后续帧的全部去噪步骤，并非逐步骤精确
   对齐（完整 Consistory / Story-Diffusion 实现需逐 step 共享）。
2. ``all-in-one`` 方案依赖 IP-Adapter 权重下载；下载失败时自动回退到
   仅 prompt 注入的一致性策略。
3. ``reference_image`` 在 ``consistory`` / ``story-diffusion`` 方法下
   仅作为一致性评分的锚点与首帧 latent 初始化参考；真正的特征锚点来
   自首帧文本生成。``all-in-one`` 方法下参考图会作为 IP-Adapter 图像
   条件注入所有帧。
4. 一致性评分使用结构相似度（SSIM / 互相关），而非 CLIP 语义相似度；
   后者需额外加载 CLIP 视觉模型，留作未来增强。
5. 复杂姿势、大角度变化、多角色场景下一致性可能不稳定，建议配合
   :class:`IdentityKeeper` / :class:`StyleKeeper` 组合使用。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.consistency._base import BaseConsistencyNode

__all__ = ["CrossFrameConsistency"]


# Story-Diffusion 默认回退到 SD 1.5 时的额外显存 / 许可证信息
_EXTRA_VRAM: Dict[str, float] = {
    "runwayml/stable-diffusion-v1-5": 4.0,
}
_EXTRA_LICENSE: Dict[str, str] = {
    "runwayml/stable-diffusion-v1-5": "OpenRAIL-M (CreativeML Open RAIL-M)",
}


@registry.register
class CrossFrameConsistency(BaseConsistencyNode):
    """跨帧一致性生成节点。

    根据一组提示词（``prompts``）逐帧生成图像，并通过角色描述
    （``character_description``）注入与方法特定的注意力 / 图像条件机制
    保持跨帧的角色与主体一致性。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认
        ``"stabilityai/stable-diffusion-xl-base-1.0"``。
        ``story-diffusion`` 方法下若传入 SDXL 模型会自动回退到
        ``runwayml/stable-diffusion-v1-5``。
    method:
        一致性方法，可选 ``"consistory"`` / ``"story-diffusion"`` /
        ``"all-in-one"``，默认 ``"consistory"``。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，可选 ``"float16"`` / ``"float32"`` / ``"bfloat16"``，
        默认 ``"float16"``。
    scheduler:
        显存调度器实例，``None`` 使用全局单例。
    bus:
        事件总线实例，``None`` 使用全局单例。
    **kwargs:
        透传给 :class:`BaseConsistencyNode` 的其它参数（如 ``name``）。

    Examples
    --------
    >>> node = CrossFrameConsistency(method="consistory")
    >>> result = node(MosaicData(
    ...     prompts=[
    ...         "a girl reading a book under a tree",
    ...         "the girl walking through a forest",
    ...         "the girl sitting by a campfire",
    ...     ],
    ...     character_description="a young girl with red hair, "
    ...                          "wearing a green dress, freckles",
    ...     width=1024, height=1024,
    ...     consistency_strength=0.85,
    ... ))
    >>> for i, img in enumerate(result["images"]):
    ...     img.save(f"frame_{i}.png")
    """

    name: str = "cross-frame-consistency"
    description: str = (
        "Generate a sequence of frames with cross-frame character/subject "
        "consistency. Supports consistory (cross-attn KV sharing), "
        "story-diffusion (self-attn sharing) and all-in-one (IP-Adapter) "
        "methods."
    )
    version: str = "0.1.0"
    input_types: List[str] = ["image", "text", "mosaic"]
    output_types: List[str] = ["image"]

    #: 支持的一致性方法集合。
    _SUPPORTED_METHODS: Tuple[str, ...] = (
        "consistory",
        "story-diffusion",
        "all-in-one",
    )

    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        method: str = "consistory",
        device: str = "cuda",
        dtype: str = "float16",
        scheduler: Optional[Any] = None,
        bus: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method {method!r}. "
                f"Expected one of {self._SUPPORTED_METHODS}."
            )
        super().__init__(
            device=device, dtype=dtype, scheduler=scheduler, bus=bus, **kwargs
        )
        self._model_name: str = model
        self._method: str = method
        # 实际加载时使用的模型（story-diffusion 可能回退到 SD 1.5）
        self._effective_model_name: str = self._resolve_effective_model()

        # 运行时状态
        self._kv_mode: str = "off"  # "off" / "anchor" / "share"
        self._consistency_strength: float = 0.85
        self._shared_kv_cache: Dict[str, Any] = {}
        self._ip_adapter_loaded: bool = False
        self._attention_wrapped: bool = False

    # ------------------------------------------------------------------
    # 模型解析
    # ------------------------------------------------------------------
    def _resolve_effective_model(self) -> str:
        """解析实际加载的模型标识。

        ``story-diffusion`` 方法基于 SD 1.5 自注意力共享，若用户传入
        SDXL 模型则自动回退到 ``runwayml/stable-diffusion-v1-5`` 并记录
        日志。
        """
        if self._method == "story-diffusion" and "xl" in self._model_name.lower():
            self._logger.info(
                "story-diffusion method requires an SD 1.x model; "
                "falling back from %s to runwayml/stable-diffusion-v1-5.",
                self._model_name,
            )
            return "runwayml/stable-diffusion-v1-5"
        return self._model_name

    def _resolve_target_device(self) -> str:
        """解析实际推理设备，无 GPU 时从调度器降级。"""
        device = self._device
        if device.startswith("cuda") and not self._scheduler.is_gpu:
            device = self._scheduler.device
            self._logger.info(
                "No GPU available; falling back to device %r.", device
            )
        return device

    # ------------------------------------------------------------------
    # 加载 / 卸载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载一致性生成 Pipeline。

        通过 ``Scheduler.track`` 注册显存跟踪，随后按 ``method`` 加载对应
        的 diffusers Pipeline 并应用显存优化与一致性扩展。
        """
        self._scheduler.track(self)

        if self._pipeline is not None:
            self._loaded = True
            return

        self._logger.info(
            "Loading CrossFrameConsistency pipeline (method=%s, model=%s) ...",
            self._method,
            self._effective_model_name,
        )

        if self._method == "consistory":
            self._load_consistory()
        elif self._method == "story-diffusion":
            self._load_story_diffusion()
        else:  # all-in-one
            self._load_all_in_one()

        self._loaded = True

    def _load_consistory(self) -> None:
        """加载 SDXL Pipeline 并启用交叉注意力 KV 共享（Consistory）。"""
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        torch_dtype = self._resolve_dtype()
        variant = "fp16" if self._dtype_str in ("float16", "fp16") else None

        self._pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._effective_model_name,
            torch_dtype=torch_dtype,
            variant=variant,
        )
        self._pipeline = self._pipeline.to(self._resolve_target_device())
        self._apply_optimizations()
        self._enable_attention_sharing(target="cross")

        self._logger.info(
            "Consistory SDXL pipeline loaded; cross-attn KV sharing enabled."
        )

    def _load_story_diffusion(self) -> None:
        """加载 SD Pipeline 并启用自注意力 KV 共享（Story-Diffusion）。"""
        from diffusers import StableDiffusionPipeline  # type: ignore

        torch_dtype = self._resolve_dtype()
        variant = "fp16" if self._dtype_str in ("float16", "fp16") else None

        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self._effective_model_name,
            torch_dtype=torch_dtype,
            variant=variant,
        )
        self._pipeline = self._pipeline.to(self._resolve_target_device())
        self._apply_optimizations()
        self._enable_attention_sharing(target="self")

        self._logger.info(
            "Story-Diffusion SD pipeline loaded; self-attn KV sharing enabled."
        )

    def _load_all_in_one(self) -> None:
        """加载 SDXL Pipeline + IP-Adapter（all-in-one 简化方案）。"""
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        torch_dtype = self._resolve_dtype()
        variant = "fp16" if self._dtype_str in ("float16", "fp16") else None

        self._pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._effective_model_name,
            torch_dtype=torch_dtype,
            variant=variant,
        )
        self._pipeline = self._pipeline.to(self._resolve_target_device())
        self._apply_optimizations()
        self._load_ip_adapter()

        self._logger.info(
            "All-in-one SDXL pipeline loaded; IP-Adapter enabled=%s.",
            self._ip_adapter_loaded,
        )

    def _load_ip_adapter(self) -> None:
        """加载 IP-Adapter 用于参考图 / 首帧驱动的跨帧一致。

        加载失败时记录警告并回退到仅 prompt 注入的一致性策略，节点仍可
        正常工作。
        """
        try:
            self._pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl.safetensors",
            )
            self._ip_adapter_loaded = True
            self._logger.info("IP-Adapter loaded for cross-frame consistency.")
        except Exception as exc:  # noqa: BLE001
            self._ip_adapter_loaded = False
            self._logger.warning(
                "Failed to load IP-Adapter: %s. "
                "Falling back to prompt-only consistency.",
                exc,
            )

    def unload(self) -> None:
        """释放 Pipeline 与一致性扩展状态。"""
        self._pipeline = None
        self._loaded = False
        self._shared_kv_cache.clear()
        self._kv_mode = "off"
        self._ip_adapter_loaded = False
        self._attention_wrapped = False
        self._logger.info(
            "CrossFrameConsistency pipeline unloaded (method=%s).",
            self._method,
        )

    # ------------------------------------------------------------------
    # 注意力 KV 共享扩展
    # ------------------------------------------------------------------
    def _enable_attention_sharing(self, target: str) -> None:
        """对 UNet 注意力模块的 K/V 投影进行包装以支持跨帧共享。

        Parameters
        ----------
        target:
            ``"cross"`` 包装交叉注意力（``attn2`` 模块，用于 Consistory）；
            ``"self"`` 包装自注意力（``attn1`` 模块，用于 Story-Diffusion）。

        Notes
        -----
        通过名称启发式（``attn1`` / ``attn2``）定位自 / 交叉注意力模块。
        若未匹配到任何模块，则记录警告并跳过（节点仍可生成，仅失去 KV
        共享一致性约束）。
        """
        unet = getattr(self._pipeline, "unet", None)
        if unet is None:
            self._logger.warning(
                "Pipeline has no UNet; attention KV sharing disabled."
            )
            return

        target_token = "attn2" if target == "cross" else "attn1"
        count = 0
        for name, module in unet.named_modules():
            if not (hasattr(module, "to_k") and hasattr(module, "to_v")):
                continue
            if target_token not in name:
                continue
            self._wrap_kv_projection(module.to_k, name, "k")
            self._wrap_kv_projection(module.to_v, name, "v")
            count += 1

        if count == 0:
            self._logger.warning(
                "No %s-attention modules (%s) found in UNet; "
                "KV sharing disabled.",
                target,
                target_token,
            )
            self._attention_wrapped = False
        else:
            self._attention_wrapped = True
            self._logger.info(
                "Enabled %s-attention KV sharing on %d module pair(s).",
                target,
                count,
            )

    def _wrap_kv_projection(
        self,
        proj: Any,
        module_name: str,
        kind: str,
    ) -> None:
        """包装一个 K/V 线性投影，使其在跨帧生成时缓存并融合输出。

        Parameters
        ----------
        proj:
            注意力模块的 ``to_k`` / ``to_v`` 子模块（``nn.Linear`` 兼容）。
        module_name:
            所属注意力模块在 UNet 中的完整名称，用作缓存键前缀。
        kind:
            ``"k"`` 或 ``"v"``，标识键 / 值投影。
        """
        original_forward = proj.forward
        cache = self._shared_kv_cache
        node = self

        def wrapped_forward(x: Any, *args: Any, **kwargs: Any) -> Any:
            out = original_forward(x, *args, **kwargs)
            key = f"{module_name}:{kind}"
            mode = node._kv_mode
            if mode == "anchor":
                # 锚帧：缓存当前 K/V 投影输出
                try:
                    cache[key] = out.detach().clone()
                except Exception:  # noqa: BLE001
                    pass
            elif mode == "share" and key in cache:
                # 后续帧：按 consistency_strength 融合锚帧 K/V
                cached = cache[key]
                try:
                    if cached.shape == out.shape:
                        ratio = node._consistency_strength
                        cached = cached.to(device=out.device, dtype=out.dtype)
                        out = (1.0 - ratio) * out + ratio * cached
                except Exception:  # noqa: BLE001
                    # 形状不匹配等异常时回退到原始输出
                    pass
            return out

        proj.forward = wrapped_forward  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # IP-Adapter 辅助
    # ------------------------------------------------------------------
    def _set_ip_adapter_scale(self, scale: float) -> None:
        """安全设置 IP-Adapter 的图像条件强度。"""
        if not self._ip_adapter_loaded:
            return
        try:
            self._pipeline.set_ip_adapter_scale(float(scale))
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("set_ip_adapter_scale failed: %s", exc)

    @staticmethod
    def _neutral_ip_image() -> Any:
        """生成一张中性白色图像，用作首帧 IP-Adapter 的占位输入。

        首帧尚无锚帧可参考时，以强度 0 传入中性图像可避免 IP-Adapter
        处理器因缺少 ``ip_adapter_image`` 而报错。
        """
        from PIL import Image  # type: ignore

        return Image.new("RGB", (224, 224), (255, 255, 255))

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------
    @staticmethod
    def _inject_character(prompt: str, character_description: str) -> str:
        """将角色描述注入单帧 prompt，确保角色一致。

        若角色描述已存在于 prompt 中则原样返回，否则以逗号追加。

        Parameters
        ----------
        prompt:
            单帧原始提示词。
        character_description:
            角色描述文本。

        Returns
        -------
        str
            注入角色描述后的完整提示词。
        """
        prompt = (prompt or "").strip()
        desc = (character_description or "").strip()
        if not desc:
            return prompt
        if desc.lower() in prompt.lower():
            return prompt
        return f"{prompt}, {desc}"

    # ------------------------------------------------------------------
    # 单帧生成
    # ------------------------------------------------------------------
    def _generate_frame(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
        ip_adapter_image: Any = None,
    ) -> Any:
        """生成单帧图像。

        Parameters
        ----------
        prompt:
            已注入角色描述的完整提示词。
        negative_prompt:
            反向提示词，``None`` 表示不使用。
        width, height:
            输出尺寸（已对齐到 8 的倍数）。
        num_inference_steps:
            推理步数。
        guidance_scale:
            引导系数。
        generator:
            ``torch.Generator``，跨帧复用以保证可复现。
        ip_adapter_image:
            IP-Adapter 图像条件（仅 ``all-in-one`` 方法使用）。

        Returns
        -------
        PIL.Image.Image
            生成的单帧图像。
        """
        pipe_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": 1,
            "generator": generator,
        }
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt
        if self._ip_adapter_loaded and ip_adapter_image is not None:
            pipe_kwargs["ip_adapter_image"] = ip_adapter_image

        output = self._run_pipeline(**pipe_kwargs)
        images = output.images if hasattr(output, "images") else []
        return images[0] if images else None

    # ------------------------------------------------------------------
    # 推理主流程
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行跨帧一致性生成。

        Parameters
        ----------
        input_data:
            必须包含：

            * ``prompts`` (list[str])：每帧的提示词，长度决定生成帧数。
            * ``character_description`` (str)：角色 / 主体描述，注入每帧。

            可选：

            * ``reference_image`` (PIL.Image 或路径)：参考锚帧。
            * ``negative_prompt`` (str)
            * ``width`` (int, 默认 1024)
            * ``height`` (int, 默认 1024)
            * ``num_inference_steps`` (int, 默认 30)
            * ``guidance_scale`` (float, 默认 7.5)
            * ``seed`` (int)
            * ``consistency_strength`` (float, 默认 0.85)

        Returns
        -------
        MosaicData
            包含 ``images`` (list[PIL.Image])、``reference_image`` (可选)、
            ``character_description`` (str)、``consistency_scores``
            (list[float])、``average_consistency`` (float)、``seed`` (int)。

        Raises
        ------
        ValueError
            缺少 ``prompts`` 或 ``character_description``，或 ``prompts``
            非非空字符串列表。
        """
        self._emit_start()
        t0 = time.perf_counter()
        try:
            # -- 校验必填输入（fail-fast，无需加载模型） ----------------
            prompts = input_data.get("prompts")
            if (
                not isinstance(prompts, list)
                or not prompts
                or not all(isinstance(p, str) for p in prompts)
            ):
                raise ValueError(
                    "CrossFrameConsistency requires 'prompts' "
                    "(non-empty list[str])."
                )

            character_description = input_data.get("character_description")
            if (
                not isinstance(character_description, str)
                or not character_description.strip()
            ):
                raise ValueError(
                    "CrossFrameConsistency requires 'character_description' "
                    "(non-empty str)."
                )

            # -- 加载模型（输入校验通过后再加载） -----------------------
            self._scheduler.ensure_loaded(self)

            # -- 解析参数 -------------------------------------------------
            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            width = int(input_data.get("width", 1024))
            height = int(input_data.get("height", 1024))
            # 对齐到 8 的倍数
            width = max(8, (width // 8) * 8)
            height = max(8, (height // 8) * 8)

            # story-diffusion (SD 1.5) 在 512 附近表现最佳
            if self._method == "story-diffusion":
                longest = max(width, height)
                if longest > 512:
                    self._logger.warning(
                        "story-diffusion (SD 1.5) performs best at 512px; "
                        "clamping from (%d, %d).",
                        width,
                        height,
                    )
                    ratio = 512.0 / longest
                    width = max(8, (int(width * ratio) // 8) * 8)
                    height = max(8, (int(height * ratio) // 8) * 8)

            num_inference_steps = int(
                input_data.get("num_inference_steps", 30)
            )
            guidance_scale = float(input_data.get("guidance_scale", 7.5))

            consistency_strength = float(
                input_data.get("consistency_strength", 0.85)
            )
            consistency_strength = max(0.0, min(1.0, consistency_strength))
            self._consistency_strength = consistency_strength

            # -- 参考图 ---------------------------------------------------
            reference_image = input_data.get("reference_image")
            if reference_image is not None:
                reference_image = self._load_image(reference_image)

            # -- 种子 -----------------------------------------------------
            seed, generator = self._prepare_seed(input_data.get("seed"))

            # -- 逐帧生成 -------------------------------------------------
            total = len(prompts)
            images: List[Any] = []

            # all-in-one: 参考图作为 IP-Adapter 锚帧
            ip_anchor = (
                reference_image
                if (self._method == "all-in-one" and reference_image is not None)
                else None
            )

            for i, base_prompt in enumerate(prompts):
                full_prompt = self._inject_character(
                    base_prompt, character_description
                )

                # KV 共享模式（consistory / story-diffusion）
                if self._attention_wrapped:
                    self._kv_mode = "anchor" if i == 0 else "share"

                # IP-Adapter 图像条件（all-in-one）
                ip_image: Any = None
                if self._method == "all-in-one" and self._ip_adapter_loaded:
                    if ip_anchor is not None:
                        ip_image = ip_anchor
                        self._set_ip_adapter_scale(consistency_strength)
                    elif i == 0:
                        # 首帧无锚帧：以强度 0 传入中性图像
                        ip_image = self._neutral_ip_image()
                        self._set_ip_adapter_scale(0.0)
                    else:
                        ip_image = images[0]
                        self._set_ip_adapter_scale(consistency_strength)

                frame = self._generate_frame(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    ip_adapter_image=ip_image,
                )

                if frame is None:
                    from PIL import Image  # type: ignore

                    frame = Image.new("RGB", (width, height), (0, 0, 0))
                    self._logger.warning(
                        "Frame %d generation returned no image; "
                        "using placeholder.",
                        i,
                    )

                images.append(frame)
                self._emit_progress(
                    current=i + 1,
                    total=total,
                    message=f"Generated frame {i + 1}/{total}",
                )

            # -- 一致性评分 -----------------------------------------------
            anchor_for_scoring = (
                reference_image
                if reference_image is not None
                else (images[0] if images else None)
            )
            consistency_scores: List[float] = []
            for img in images:
                if anchor_for_scoring is None or img is None:
                    consistency_scores.append(0.0)
                else:
                    consistency_scores.append(
                        self._compute_image_similarity(anchor_for_scoring, img)
                    )
            average_consistency = (
                sum(consistency_scores) / len(consistency_scores)
                if consistency_scores
                else 0.0
            )
        except Exception as exc:
            self._emit_error(exc)
            raise
        finally:
            # 确保异常路径下也重置 KV 共享模式
            self._kv_mode = "off"

        elapsed = time.perf_counter() - t0

        result = MosaicData(
            images=images,
            reference_image=reference_image,
            character_description=character_description,
            consistency_scores=consistency_scores,
            average_consistency=average_consistency,
            seed=seed,
            model_name=self._effective_model_name,
            method=self._method,
            num_frames=len(images),
            width=width,
            height=height,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "num_frames": len(images),
                "method": self._method,
                "average_consistency": average_consistency,
                "consistency_strength": consistency_strength,
                "seed": seed,
                "width": width,
                "height": height,
            },
        )
        return result

    # ------------------------------------------------------------------
    # 节点规格
    # ------------------------------------------------------------------
    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息与一致性方法。"""
        info = self._build_model_info(self._effective_model_name)

        # 补充 SD 1.5 回退模型的显存 / 许可证
        if self._effective_model_name in _EXTRA_VRAM:
            info["vram_gb"] = _EXTRA_VRAM[self._effective_model_name]
        if self._effective_model_name in _EXTRA_LICENSE:
            info["license"] = _EXTRA_LICENSE[self._effective_model_name]

        info["method"] = self._method
        info["supported_methods"] = list(self._SUPPORTED_METHODS)
        info["consistency_strength_default"] = 0.85
        info["configured_model"] = self._model_name

        if self._method == "all-in-one":
            # IP-Adapter 额外显存
            info["vram_gb"] = float(info.get("vram_gb", 8.0)) + 2.0
            info["ip_adapter"] = self._ip_adapter_loaded
        elif self._method == "consistory":
            info["kv_sharing"] = "cross-attention"
        elif self._method == "story-diffusion":
            info["kv_sharing"] = "self-attention"

        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info=info,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"method={self._method!r} model={self._effective_model_name!r} "
            f"state={status}>"
        )
