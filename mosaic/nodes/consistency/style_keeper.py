# mosaic/nodes/consistency/style_keeper.py
"""StyleKeeper 风格保持节点。

根据参考图像生成保持其风格（色调、纹理、画派）的新图像。
支持三种风格保持方法：

* ``"ip-adapter"``    —— 基于 IP-Adapter，通过图像提示注入风格嵌入，
  通用且可控。
* ``"style-aligned"`` —— 基于 StyleAligned，通过共享自注意力在批量内
  传播参考图的风格特征，无需额外权重。
* ``"reference-only"``—— 基于 Reference-Only，通过共享自注意力将参考图
  特征注入目标生成（SD1.5 基础模型），是经典 ControlNet 参考模式的
  注意力注入实现。

设计要点
--------
* 继承 :class:`BaseConsistencyNode`，复用其图像前后处理、显存优化、事件发射与
  随机种子准备逻辑。
* ``diffusers`` / ``torch`` / ``PIL`` 全部惰性导入，使本模块在依赖缺失时仍可
  被注册表发现与导入。
* ``style-aligned`` 与 ``reference-only`` 通过自定义自注意力处理器实现风格共享
  （将参考图与目标图置于同一去噪批量，并在自注意力层共享 K/V），这是两种方法
  的标准实现思路；``style_strength`` 控制共享特征与自身特征的混合比例。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。
"""

from __future__ import annotations

import time
from typing import Any

from mosaic.core.device_utils import upcast_pipeline_components
from mosaic.core.node import NodeSpec
from mosaic.core.registry import registry
from mosaic.core.types import MosaicData

from mosaic.nodes.coerce import safe_float, safe_int
from mosaic.nodes.consistency._base import BaseConsistencyNode

__all__ = ["StyleKeeper"]


# 支持的风格保持方法
_SUPPORTED_METHODS: tuple[str, ...] = (
    "ip-adapter",
    "style-aligned",
    "reference-only",
)

# IP-Adapter 权重（SDXL，全局风格）
_IP_ADAPTER_REPO = "h94/IP-Adapter"
_IP_ADAPTER_SUBFOLDER = "sdxl_models"
_IP_ADAPTER_WEIGHT = "ip-adapter_sdxl_vit-h.safetensors"

# 各 method 的默认基础模型
_SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
_SD15_BASE = "runwayml/stable-diffusion-v1-5"


# ---------------------------------------------------------------------------
# 共享自注意力处理器（StyleAligned / Reference-Only 通用）
# ---------------------------------------------------------------------------
class _SharedSelfAttnProcessor2_0:
    """共享自注意力处理器（diffusers 2.0 后端，PyTorch SDPA）。

    在自注意力层（``attn1``）中，将批量内所有元素的 K/V 汇聚为一份"共享
    特征库"，每个查询同时关注自身 K/V 与共享 K/V，并按 ``scale`` 混合两者
    的注意力输出。这使得同一去噪批量内的参考图与目标图共享风格特征。

    Parameters
    ----------
    scale:
        共享特征混合比例，``0.0`` 退化为普通自注意力，``1.0`` 完全使用共享
        特征。对应 :class:`StyleKeeper` 的 ``style_strength``。
    """

    def __init__(self, scale: float = 0.7) -> None:
        self.scale = float(max(0.0, min(1.0, scale)))

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        temb: Any | None = None,
        scale: float | None = None,
        **kwargs: Any,
    ) -> Any:
        import torch  # type: ignore

        # 是否为自注意力（attn1）：encoder_hidden_states 为 None
        is_self_attn = encoder_hidden_states is None
        mix_scale = self.scale if scale is None else float(scale)

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # ---- 风格共享：仅在自注意力层且批量大于 1 时生效 ----
        do_share = is_self_attn and batch_size > 1 and mix_scale > 0.0
        if do_share:
            # 跨批量汇聚共享 K/V：每个查询关注全部批量的 K/V
            shared_key = key.reshape(
                1, attn.heads, -1, head_dim
            ).expand(batch_size, -1, -1, -1)
            shared_value = value.reshape(
                1, attn.heads, -1, head_dim
            ).expand(batch_size, -1, -1, -1)

            out_own = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False,
            )
            out_shared = torch.nn.functional.scaled_dot_product_attention(
                query, shared_key, shared_value, attn_mask=None,
                dropout_p=0.0, is_causal=False,
            )
            hidden_states = (
                1.0 - mix_scale
            ) * out_own + mix_scale * out_shared
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False,
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                b, c, h, w
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


@registry.register
class StyleKeeper(BaseConsistencyNode):
    """风格保持节点：根据参考图生成保持其风格的新图像。

    Parameters
    ----------
    model:
        HuggingFace 模型标识或本地路径。含义随 ``method`` 变化：

        * ``"ip-adapter"``    -> IP-Adapter 权重仓库（默认 ``"h94/IP-Adapter"``），
          基础 SDXL 模型使用 ``"stabilityai/stable-diffusion-xl-base-1.0"``。
        * ``"style-aligned"`` -> SDXL 基础模型（默认
          ``"stabilityai/stable-diffusion-xl-base-1.0"``）。
        * ``"reference-only"``-> SD1.5 基础模型（默认
          ``"runwayml/stable-diffusion-v1-5"``）。
    method:
        风格保持方法，可选 ``"ip-adapter"`` / ``"style-aligned"`` /
        ``"reference-only"``，默认 ``"ip-adapter"``。
    device:
        推理设备，默认 ``"cuda"``；无 GPU 时自动降级为 ``"cpu"``。
    dtype:
        推理精度，可选 ``"float16"`` / ``"float32"`` / ``"bfloat16"``，
        默认 ``"float16"``。
    **kwargs:
        透传给 :class:`BaseConsistencyNode` 的参数（``scheduler`` / ``bus`` 等）。

    Limitations
    -----------
    * 需要 ``diffusers``、``torch``、``Pillow``。
    * ``style-aligned`` 与 ``reference-only`` 通过共享自注意力实现，参考图被
      编码为潜变量并与目标图置于同一去噪批量；``style_strength`` 控制共享特征
      的混合比例。两者均无需额外权重文件。
    * ``reference-only`` 实质是 Reference-Only 的注意力注入实现（非独立
      ControlNet 权重），与 SD1.5 基础模型配合使用。
    * 输出中的 ``style`` 字段为可选的风格描述，当前返回 ``None``，可由下游
      图像描述节点填充。
    * GPU 强烈推荐；CPU 模式下推理极慢。

    Examples
    --------
    >>> keeper = StyleKeeper(method="ip-adapter")
    >>> result = keeper(MosaicData(
    ...     reference_image="style_ref.png",
    ...     prompt="a cat sitting on a windowsill",
    ...     style_strength=0.7,
    ...     seed=42,
    ... ))
    >>> result["image"].save("styled_cat.png")
    """

    name: str = "style-keeper"
    description: str = (
        "Generate style-preserving images from a reference image using "
        "IP-Adapter, StyleAligned, or Reference-Only attention sharing. "
        "style_strength controls the degree of style transfer."
    )
    version: str = "0.1.0"
    input_types: tuple[str, ...] = ("image", "mosaic")
    output_types: tuple[str, ...] = ("image",)

    def __init__(
        self,
        model: str = "h94/IP-Adapter",
        method: str = "ip-adapter",
        device: str = "cuda",
        dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method {method!r}. "
                f"Choose from {_SUPPORTED_METHODS}."
            )
        # 先解析基础模型，用于 auto_resolve_device_dtype 检测 SD 1.5
        self._model_name: str = model
        self._base_model: str = self._resolve_base_model(model, method)
        super().__init__(device=device, dtype=dtype, model=self._base_model, **kwargs)
        self._method: str = method
        # 运行时：保存原始注意力处理器以便恢复
        self._orig_attn_procs: dict[str, Any] | None = None

    @staticmethod
    def _resolve_base_model(model: str, method: str) -> str:
        """根据 method 解析基础扩散模型标识。"""
        if method == "ip-adapter":
            # model 指 IP-Adapter 权重仓库；基础模型固定为 SDXL
            return _SDXL_BASE
        if method == "style-aligned":
            # model 应为 SDXL 基础模型；若用户保留了默认 IP-Adapter 仓库，
            # 则回退到 SDXL 基础模型
            if model == "h94/IP-Adapter":
                return _SDXL_BASE
            return model
        # reference-only：SD1.5 基础模型
        if model == "h94/IP-Adapter":
            return _SD15_BASE
        return model

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载风格保持 Pipeline。

        通过 :meth:`Scheduler.track` 注册显存跟踪后，根据 ``method`` 分发到
        具体的加载器。本方法由 ``Scheduler.ensure_loaded`` 回调，不应在其中
        调用 ``ensure_loaded`` 以免递归。
        """
        self._scheduler.track(self)

        if self._pipeline is not None:
            self._loaded = True
            return

        self._logger.info(
            "Loading StyleKeeper pipeline (method=%s, base=%s) ...",
            self._method,
            self._base_model,
        )

        if self._method == "ip-adapter":
            self._load_ip_adapter()
        elif self._method == "style-aligned":
            self._load_style_aligned()
        else:  # reference-only
            self._load_reference_only()

        self._apply_optimizations()
        # SD 1.5 (reference-only) 的 VAE 在 float16 下会产生 NaN → 黑图；
        # SDXL (ip-adapter / style-aligned) 的 VAE 已兼容 fp16，upcast 幂等。
        upcast_pipeline_components(self._pipeline, self._model_name, self._logger)
        self._loaded = True

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

    def _load_ip_adapter(self) -> None:
        """加载 SDXL Pipeline 并挂载 IP-Adapter（全局风格）权重。"""
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        from mosaic.nodes._model_loader import safe_load_pipeline

        # 走 safe_load_pipeline：统一 model_cache 缓存、fp16 variant 回退、
        # cache_dir 解析与版本诊断，避免直接调用具体 Pipeline 类的 from_pretrained。
        torch_dtype = self._resolve_dtype()
        self._pipeline = safe_load_pipeline(
            StableDiffusionXLPipeline,
            self._base_model,
            torch_dtype=torch_dtype,
            dtype_str=self._dtype_str,
        )
        self._move_pipeline_to_device()

        try:
            self._pipeline.load_ip_adapter(
                self._model_name,
                subfolder=_IP_ADAPTER_SUBFOLDER,
                weight_name=_IP_ADAPTER_WEIGHT,
            )
            self._logger.info(
                "IP-Adapter weights loaded from %s/%s.",
                self._model_name,
                _IP_ADAPTER_WEIGHT,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to load IP-Adapter weights (%s): %s. "
                "Style transfer will be disabled.",
                self._model_name,
                exc,
            )

    def _load_style_aligned(self) -> None:
        """加载普通 SDXL Pipeline（StyleAligned 无需额外权重）。"""
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        from mosaic.nodes._model_loader import safe_load_pipeline

        # 走 safe_load_pipeline：统一 model_cache 缓存、fp16 variant 回退、
        # cache_dir 解析与版本诊断，避免直接调用具体 Pipeline 类的 from_pretrained。
        torch_dtype = self._resolve_dtype()
        self._pipeline = safe_load_pipeline(
            StableDiffusionXLPipeline,
            self._base_model,
            torch_dtype=torch_dtype,
            dtype_str=self._dtype_str,
        )
        self._move_pipeline_to_device()
        self._logger.info(
            "SDXL pipeline loaded for StyleAligned (dtype=%s, device=%s).",
            self._dtype_str,
            self._device,
        )

    def _load_reference_only(self) -> None:
        """加载普通 SD1.5 Pipeline（Reference-Only 经由注意力注入实现）。"""
        from diffusers import StableDiffusionPipeline  # type: ignore

        from mosaic.nodes._model_loader import safe_load_pipeline

        # 走 safe_load_pipeline：统一 model_cache 缓存、fp16 variant 回退、
        # cache_dir 解析与版本诊断，避免直接调用具体 Pipeline 类的 from_pretrained。
        torch_dtype = self._resolve_dtype()
        self._pipeline = safe_load_pipeline(
            StableDiffusionPipeline,
            self._base_model,
            torch_dtype=torch_dtype,
            dtype_str=self._dtype_str,
        )
        self._move_pipeline_to_device()
        self._logger.info(
            "SD1.5 pipeline loaded for Reference-Only "
            "(dtype=%s, device=%s).",
            self._dtype_str,
            self._device,
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行风格保持生成。

        Parameters
        ----------
        input_data:
            必须包含 ``reference_image`` (``PIL.Image`` 或文件路径 str) 与
            ``prompt`` (str)；可选 ``negative_prompt`` (str)、``width``
            (int, 默认 1024)、``height`` (int, 默认 1024)、
            ``num_inference_steps`` (int, 默认 30)、``guidance_scale``
            (float, 默认 7.5)、``style_strength`` (float, 默认 0.7)、
            ``seed`` (int)。

        Returns
        -------
        MosaicData
            包含 ``image`` (PIL.Image)、``reference_image`` (PIL.Image)、
            ``style`` (str | None，当前为 None)、``seed`` (int)、
            ``method`` (str)、``model_name`` (str)。

        Raises
        ------
        ValueError
            缺少 ``reference_image`` / ``prompt``。
        """
        self._scheduler.ensure_loaded(self)

        self._emit_start()
        t0 = time.perf_counter()
        try:
            # ---------- 校验输入 ----------
            reference_image = input_data.get("reference_image")
            if reference_image is None:
                raise ValueError(
                    "StyleKeeper requires 'reference_image' "
                    "(PIL.Image or file path)."
                )
            prompt = input_data.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(
                    f"StyleKeeper requires 'prompt' (str), "
                    f"got {type(prompt).__name__}."
                )

            negative_prompt = input_data.get("negative_prompt")
            if not isinstance(negative_prompt, str):
                negative_prompt = None

            # ---------- 提取参数 ----------
            width = safe_int(input_data.get("width"), "width", default=1024)
            height = safe_int(input_data.get("height"), "height", default=1024)
            width = max(8, (width // 8) * 8)
            height = max(8, (height // 8) * 8)
            # 大图像内存保护
            self._check_image_dimensions(width, height)

            num_inference_steps = safe_int(
                input_data.get("num_inference_steps"),
                "num_inference_steps",
                default=30,
            )
            guidance_scale = safe_float(
                input_data.get("guidance_scale"), "guidance_scale", default=7.5
            )
            style_strength = safe_float(
                input_data.get("style_strength"), "style_strength", default=0.7
            )
            style_strength = max(0.0, min(1.0, style_strength))

            # ---------- 图像前处理 ----------
            # 参考图统一 resize 到 512x512
            ref_img = self._load_image(reference_image)
            ref_512 = self._resize_to_model(ref_img, (512, 512))

            # ---------- 种子 ----------
            seed, generator = self._prepare_seed(input_data.get("seed"))

            # ---------- 分发推理 ----------
            self._emit_progress(
                current=0, total=1, message="Generating style-preserving image"
            )
            if self._method == "ip-adapter":
                image = self._generate_ip_adapter(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ref_512=ref_512,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    style_strength=style_strength,
                    generator=generator,
                )
            elif self._method == "style-aligned":
                image = self._generate_style_aligned(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ref_img=ref_512,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    style_strength=style_strength,
                    generator=generator,
                )
            else:  # reference-only
                image = self._generate_reference_only(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    ref_img=ref_512,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    style_strength=style_strength,
                    generator=generator,
                )
            self._emit_progress(
                current=1, total=1, message="Generation complete"
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_error(exc)
            raise

        elapsed = time.perf_counter() - t0

        result = MosaicData(
            image=image,
            reference_image=ref_512,
            style=None,  # 可选：由下游图像描述节点填充
            seed=seed,
            method=self._method,
            model_name=self._model_name,
        )
        self._emit_complete(
            duration=elapsed,
            output_summary={
                "seed": seed,
                "method": self._method,
                "style_strength": style_strength,
                "width": width,
                "height": height,
            },
        )
        return result

    def _generate_ip_adapter(
        self,
        prompt: str,
        negative_prompt: str | None,
        ref_512: Any,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        style_strength: float,
        generator: Any,
    ) -> Any:
        """IP-Adapter 方法推理。``style_strength`` 映射到 ip_adapter_scale。"""
        self._set_ip_adapter_scale(style_strength)

        pipe_kwargs: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "ip_adapter_image": ref_512,
        }
        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = negative_prompt

        output = self._run_pipeline(**pipe_kwargs)
        return self._extract_image(output)

    def _generate_style_aligned(
        self,
        prompt: str,
        negative_prompt: str | None,
        ref_img: Any,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        style_strength: float,
        generator: Any,
    ) -> Any:
        """StyleAligned 方法推理。

        将参考图编码为潜变量并与目标图置于同一去噪批量，在自注意力层共享
        K/V 以传播风格。``style_strength`` 控制共享特征的混合比例。
        """
        try:
            return self._generate_with_shared_attention(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ref_img=ref_img,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                style_strength=style_strength,
                generator=generator,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "StyleAligned shared-attention failed (%s); "
                "falling back to plain generation.", exc,
            )
            self._disable_shared_attention()
            return self._plain_generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

    def _generate_reference_only(
        self,
        prompt: str,
        negative_prompt: str | None,
        ref_img: Any,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        style_strength: float,
        generator: Any,
    ) -> Any:
        """Reference-Only 方法推理（SD1.5 + 共享自注意力）。"""
        try:
            return self._generate_with_shared_attention(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ref_img=ref_img,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                style_strength=style_strength,
                generator=generator,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Reference-Only shared-attention failed (%s); "
                "falling back to plain generation.", exc,
            )
            self._disable_shared_attention()
            return self._plain_generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

    # ------------------------------------------------------------------
    # 共享注意力核心
    # ------------------------------------------------------------------
    def _generate_with_shared_attention(
        self,
        prompt: str,
        negative_prompt: str | None,
        ref_img: Any,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        style_strength: float,
        generator: Any,
    ) -> Any:
        """以共享自注意力方式生成：参考图与目标图同批量去噪。"""
        import torch  # type: ignore

        pipe = self._pipeline
        device = self._infer_device()
        dtype = self._resolve_dtype()

        # 1. 参考图 resize 到输出尺寸并编码为潜变量
        ref_for_vae = self._resize_to_model(ref_img, (width, height))
        ref_latents = self._encode_image_to_latents(ref_for_vae, device, dtype)

        # 2. 构造初始潜变量批量 [ref_noisy, target_noise]
        #    参考图加噪到起始时间步，目标图使用随机噪声
        scheduler = pipe.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        init_timestep = scheduler.timesteps[0]

        ref_noise = torch.randn(
            ref_latents.shape, generator=generator, device=device, dtype=dtype
        )
        ref_noisy = scheduler.add_noise(ref_latents, ref_noise, init_timestep)

        target_noise = torch.randn(
            ref_latents.shape, generator=generator, device=device, dtype=dtype
        )
        # 批量：[参考, 目标]
        latents = torch.cat([ref_noisy, target_noise], dim=0)

        # 3. 装配共享自注意力处理器
        self._enable_shared_attention(style_strength)

        try:
            # 4. 以批量方式运行 Pipeline
            pipe_kwargs: dict = {
                "prompt": [prompt, prompt],
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "latents": latents,
                "num_images_per_prompt": 1,
            }
            if negative_prompt is not None:
                pipe_kwargs["negative_prompt"] = [negative_prompt, negative_prompt]

            output = self._run_pipeline(**pipe_kwargs)
        finally:
            self._disable_shared_attention()

        images = getattr(output, "images", None) or (
            output.get("images") if isinstance(output, dict) else None
        )
        if not images:
            raise RuntimeError("Pipeline output did not contain any image.")
        # 取目标图（批量中第二张）
        return images[1] if len(images) > 1 else images[0]

    def _enable_shared_attention(self, scale: float) -> None:
        """将 UNet 的自注意力处理器替换为共享版本。"""
        if self._pipeline is None:
            return
        unet = getattr(self._pipeline, "unet", None)
        if unet is None or not hasattr(unet, "attn_processors"):
            return
        self._orig_attn_procs = dict(unet.attn_processors)
        new_procs: dict[str, Any] = {}
        for name, proc in self._orig_attn_procs.items():
            if name.endswith("attn1.processor"):
                new_procs[name] = _SharedSelfAttnProcessor2_0(scale=scale)
            else:
                new_procs[name] = proc
        try:
            unet.set_attn_processor(new_procs)
            self._logger.debug(
                "Enabled shared self-attention (scale=%.3f).", scale
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to set shared attention: %s", exc)

    def _disable_shared_attention(self) -> None:
        """恢复 UNet 原始注意力处理器。"""
        if self._orig_attn_procs is None or self._pipeline is None:
            return
        unet = getattr(self._pipeline, "unet", None)
        if unet is not None and hasattr(unet, "set_attn_processor"):
            try:
                unet.set_attn_processor(self._orig_attn_procs)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Failed to restore attention: %s", exc)
        self._orig_attn_procs = None

    def _encode_image_to_latents(
        self, image: Any, device: str, dtype: Any
    ) -> Any:
        """将 PIL 图像通过 VAE 编码为潜变量。"""
        import torch  # type: ignore
        import numpy as np  # type: ignore

        from PIL import Image  # type: ignore

        if not isinstance(image, Image.Image):
            image = self._load_image(image)

        vae = getattr(self._pipeline, "vae", None)
        if vae is None:
            raise RuntimeError("Pipeline has no VAE for image encoding.")

        # VAE 可能被 upcast_pipeline_components 上转为 float32（防黑图），
        # 也可能保持 pipeline 的原始 dtype（如 float16）。
        # 动态获取 VAE 的实际 dtype 以确保 tensor 匹配，
        # 否则触发 "mat1 and mat2 must have the same dtype"。
        vae_dtype = next(vae.parameters()).dtype

        arr = np.array(image.convert("RGB")).astype(np.float32) / 127.5 - 1.0
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=vae_dtype)

        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        with torch.no_grad():
            latent_dist = vae.encode(tensor).latent_dist
            latents = latent_dist.sample()
            latents = latents * scaling_factor
        # 输出 latent 转回 pipeline dtype，保持与后续 UNet 推理一致
        latents = latents.to(dtype=dtype)
        return latents

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

    def _plain_generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Any:
        """回退：不使用风格共享的普通生成。"""
        pipe_kwargs: dict = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = negative_prompt
        output = self._run_pipeline(**pipe_kwargs)
        return self._extract_image(output)

    @staticmethod
    def _extract_image(output: Any) -> Any:
        """从 Pipeline 输出中提取首张图片。"""
        images = getattr(output, "images", None)
        if images:
            return images[0]
        if isinstance(output, dict) and output.get("images"):
            return output["images"][0]
        raise RuntimeError("Pipeline output did not contain any image.")

    # ------------------------------------------------------------------
    # 卸载 / 规格
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """释放 Pipeline 资源。"""
        if self._pipeline is not None:
            from mosaic.core.model_cache import model_cache

            # 优先使用加载时附加的缓存键（精确匹配），回退到运行时类型推断
            cache_cls = getattr(
                self._pipeline, "_mosaic_cache_cls", None
            )
            if not isinstance(cache_cls, str):
                cache_cls = type(self._pipeline)
            cache_dtype = getattr(
                self._pipeline, "_mosaic_cache_dtype", None
            )
            if not isinstance(cache_dtype, str):
                cache_dtype = "default"
            cache_device = getattr(
                self._pipeline, "_mosaic_cache_device", None
            )
            if not isinstance(cache_device, str):
                cache_device = None
            released = model_cache.remove(
                cache_cls,
                self._model_name,
                cache_dtype,
                cache_device,
            )
            if released:
                try:
                    self._pipeline.to("cpu")
                except Exception:
                    pass
            self._pipeline = None
            import gc

            gc.collect()
            from mosaic.core.device_utils import empty_device_cache

            empty_device_cache()
        self._orig_attn_procs = None
        self._loaded = False
        self._logger.info(
            "StyleKeeper pipeline unloaded (method=%s).", self._method
        )

    def describe(self) -> NodeSpec:
        """返回节点规格说明，含模型信息（VRAM、许可证、方法）。"""
        # model_info 以基础模型为基准估算显存与许可证
        model_info = self._build_model_info(self._base_model)
        model_info["method"] = self._method
        model_info["ip_adapter_repo"] = (
            self._model_name if self._method == "ip-adapter" else None
        )
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
            f"<StyleKeeper name={self.name!r} method={self._method!r} "
            f"base={self._base_model!r} state={status}>"
        )
