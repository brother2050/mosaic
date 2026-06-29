# mosaic/nodes/text/_base.py
"""文本域节点基类。

提取 :class:`TextGenerator` 与 :class:`Chat` 共用的模型加载/卸载与文本
生成逻辑。子类只需实现 :meth:`BaseTextNode.run` 中"如何构造输入消息"与
"如何解析输出"的部分，底层推理流程由本基类提供。

设计要点
--------
* ``transformers`` / ``torch`` 采用惰性导入，使本模块在未安装这些依赖时
  仍可被注册表发现与导入（仅在实际加载/推理时才报依赖缺失）。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理：
  ``load`` 调用 ``scheduler.track(self)`` 注册显存跟踪并执行实际加载；
  ``run`` 调用 ``scheduler.ensure_loaded(self)`` 触发按需加载 + LRU 淘汰。
  注意：``load`` 不能调用 ``ensure_loaded``（会递归，因 ``ensure_loaded``
  内部会回调 ``node.load()``）。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出 ``node_start``/
  ``node_complete``/``node_error`` 事件，回调异常被总线隔离不影响运行。
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData

__all__ = ["BaseTextNode"]


# 常见模型的粗略显存估算（fp16，GB），用于 describe() 与调度器
# 数值 = 模型权重大小（FP16/BF16），不含 KV cache 与激活内存
_VRAM_ESTIMATES: dict[str, float] = {
    "Qwen/Qwen2.5-7B-Instruct": 15.0,
    "Qwen/Qwen2.5-14B-Instruct": 28.0,
    "Qwen/Qwen2.5-72B-Instruct": 145.0,
    "meta-llama/Llama-3.1-8B-Instruct": 16.0,
    "meta-llama/Llama-3.1-70B-Instruct": 140.0,
}


class BaseTextNode(Node):
    """文本域节点抽象基类。

    封装基于 ``transformers`` 的因果语言模型加载与生成流程。子类需实现
    :meth:`run`，并通过类属性声明 ``name``/``description``/
    ``input_types``/``output_types``。

    Parameters
    ----------
    model:
        HuggingFace 模型标识，默认 ``"Qwen/Qwen2.5-7B-Instruct"``。
    device_map:
        传递给 ``from_pretrained`` 的 ``device_map``，默认 ``"auto"``。
    torch_dtype:
        权重精度，可选 ``"fp32"``/``"fp16"``/``"bf16"``，默认 ``"fp16"``。
        传递给 ``from_pretrained`` 的 ``dtype`` 参数。
    trust_remote_code:
        是否信任远程代码，默认 ``True``（Qwen 等模型需要）。
    scheduler:
        显存调度器实例，``None`` 使用全局单例。
    bus:
        事件总线实例，``None`` 使用全局单例。
    """

    domain: str = "text"
    description: str = "Base text node."
    version: str = "0.1.0"
    input_types: list[str] = ["text"]
    output_types: list[str] = ["text"]

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "fp16",
        trust_remote_code: bool = True,
        scheduler: Scheduler | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model_name: str = model
        self._device_map: str = device_map
        self._torch_dtype: str = torch_dtype
        self._trust_remote_code: bool = trust_remote_code
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._bus: EventBus = bus or get_event_bus()
        self._logger = logging.getLogger(f"mosaic.nodes.text.{self.name}")

        # 运行时持有的模型与 tokenizer（load 后填充）
        self._model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # 模型加载 / 卸载
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载模型与 tokenizer 到 GPU/CPU。

        通过 ``Scheduler.track`` 注册显存跟踪后执行实际加载。本方法由
        ``Scheduler.ensure_loaded`` 回调，不应在其中调用 ``ensure_loaded``
        以免递归。
        """
        # 注册到调度器以跟踪显存
        self._scheduler.track(self)

        if self._model is not None and self._tokenizer is not None:
            self._loaded = True
            return

        self._logger.info("Loading model %s ...", self._model_name)
        self._load_model()
        self._loaded = True

    def _load_model(self) -> None:
        """实际加载 transformers 模型与 tokenizer。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        # 加载 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
        )
        # 处理无 pad_token 的情况
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 解析精度
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(self._torch_dtype, torch.float16)

        # 加载模型（transformers 4.17+ 推荐 dtype= 替代 torch_dtype=）
        load_kwargs: dict = {
            "device_map": self._device_map,
            "trust_remote_code": self._trust_remote_code,
        }
        # 优先使用 dtype 参数（新版 transformers），回退 torch_dtype（旧版兼容）
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                dtype=resolved_dtype,
                **load_kwargs,
            )
        except TypeError:
            # 旧版 transformers 不支持 dtype 参数
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=resolved_dtype,
                **load_kwargs,
            )
        self._model.eval()

        self._logger.info(
            "Model %s loaded (dtype=%s, device_map=%s).",
            self._model_name,
            self._torch_dtype,
            self._device_map,
        )

    def unload(self) -> None:
        """释放模型与 tokenizer。

        本方法执行实际资源清理（清空模型/tokenizer 引用）。它由
        ``Scheduler.release`` / ``Scheduler._evict`` 回调，不应在其中调用
        ``scheduler.release(self)`` 以免递归。如需通过调度器释放显存，
        请直接调用 ``scheduler.release(self)``。
        """
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._logger.info("Model %s unloaded.", self._model_name)

    # ------------------------------------------------------------------
    # 公共生成逻辑
    # ------------------------------------------------------------------
    def _generate_from_messages(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> tuple[str, int, int]:
        """从对话消息列表生成文本（公共推理流程）。

        使用 ``tokenizer.apply_chat_template`` 构造输入，在
        ``torch.inference_mode()`` 下调用 ``model.generate``，解码仅新生成
        的 token。

        Parameters
        ----------
        messages:
            对话消息列表，格式 ``[{"role": str, "content": str}, ...]``。
        max_new_tokens:
            最大生成 token 数。
        temperature:
            采样温度。
        top_p:
            nucleus sampling 概率阈值。
        do_sample:
            是否采样；``False`` 时退化为贪心解码。

        Returns
        -------
        tuple[str, int, int]
            ``(generated_text, input_tokens, output_tokens)``。
        """
        import torch  # type: ignore

        # 构造输入：优先使用 chat template（返回 PyTorch tensor）
        input_ids = self._apply_chat_template(messages)

        # 确保 2D: (batch=1, seq_len)
        if hasattr(input_ids, "dim") and callable(getattr(input_ids, "dim", None)):
            try:
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
            except (TypeError, RuntimeError):
                pass

        # 迁移到模型所在设备
        model_device = self._infer_device()
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(model_device)
        input_length = input_ids.shape[-1] if hasattr(input_ids, "shape") else 0

        # 构造 attention_mask（全 1，与 input_ids 同设备）
        try:
            attention_mask = torch.ones_like(input_ids)
        except Exception:  # noqa: BLE001
            attention_mask = None

        # 生成
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        # 仅解码新生成的部分
        new_tokens = output_ids[0, input_length:]
        generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_tokens = int(new_tokens.shape[-1])

        return generated_text, int(input_length), output_tokens

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> Any:
        """应用 tokenizer 的 chat template 构造输入张量。

        使用业界推荐的两步模式（transformers 4.40+ 最佳实践）：
        1. ``apply_chat_template(tokenize=False)`` 获取格式化字符串
        2. ``tokenizer(formatted, return_tensors="pt")`` 独立 tokenize

        若 tokenizer 不支持 ``apply_chat_template``，回退为直接编码纯文本。

        Returns
        -------
        Any
            ``torch.Tensor`` 形状的 token id 张量 ``(batch, seq_len)``。
        """
        if hasattr(self._tokenizer, "apply_chat_template"):
            # Step 1: 格式化为字符串（不 tokenize，便于调试与兼容）
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Step 2: 独立 tokenize 为 PyTorch 张量
            model_inputs = self._tokenizer(formatted, return_tensors="pt")
            return model_inputs["input_ids"]

        # 回退：拼接所有 content
        flat = "\n".join(m.get("content", "") for m in messages)
        encoded = self._tokenizer(flat, return_tensors="pt")
        return encoded["input_ids"] if isinstance(encoded, dict) else encoded

    def _infer_device(self) -> str:
        """推断模型所在设备。"""
        if self._model is None:
            return self._scheduler.device
        # device_map="auto" 时模型可能跨设备，取第一个参数的设备
        try:
            return next(self._model.parameters()).device
        except (StopIteration, AttributeError):
            return self._scheduler.device

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

    # ------------------------------------------------------------------
    # Node 抽象方法（子类必须实现 run；describe 可由子类覆写）
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
        vram = _VRAM_ESTIMATES.get(self._model_name, 16.0)
        return {
            "name": self._model_name,
            "source": "HuggingFace",
            "license": "See model card on HuggingFace",
            "vram_gb": vram,
            "dtype": self._torch_dtype,
            "device_map": self._device_map,
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"model={self._model_name!r} state={status}>"
        )
