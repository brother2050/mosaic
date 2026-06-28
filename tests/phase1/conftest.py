# tests/phase1/conftest.py
"""Phase 1 测试公共 fixtures。

提供 mock 节点、测试数据等共用 fixture，供所有测试文件使用。
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.core import (
    Context,
    EventBus,
    MosaicData,
    Node,
    NodeSpec,
    Scheduler,
    TextData,
    registry,
)


# ---------------------------------------------------------------------------
# Mock torch 模块注入（避免测试环境无 torch 时导入失败）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _mock_torch_module():
    """注入 mock torch 模块，确保无 GPU 环境也可运行测试。"""
    if "torch" not in sys.modules:
        _mt = types.ModuleType("torch")
        _ctx = MagicMock()
        _ctx.__enter__ = MagicMock(return_value=None)
        _ctx.__exit__ = MagicMock(return_value=None)
        _mt.inference_mode = MagicMock(return_value=_ctx)
        _mt.float16 = "fp16"
        _mt.float32 = "fp32"
        _mt.bfloat16 = "bf16"
        _mt.Generator = MagicMock
        _mt.Tensor = MagicMock
        _mt.ones_like = MagicMock(return_value=MagicMock())
        _mt.ones = MagicMock(return_value=MagicMock())
        _mt.tensor = MagicMock(return_value=MagicMock())
        _mcuda = types.ModuleType("torch.cuda")
        _mcuda.is_available = MagicMock(return_value=False)
        _mcuda.get_device_properties = MagicMock()
        _mcuda.memory_allocated = MagicMock(return_value=0)
        _mt.cuda = _mcuda
        sys.modules["torch"] = _mt
        sys.modules["torch.cuda"] = _mcuda
    yield


# ---------------------------------------------------------------------------
# Mock Node 子类
# ---------------------------------------------------------------------------
class _MockNodeImpl(Node):
    """一个可直接实例化的 mock 节点，用于测试框架核心。"""

    name = "mock-node"
    domain = "text"
    description = "A mock node for testing."
    version = "0.1.0"
    input_types = ["text"]
    output_types = ["text"]

    def __init__(self, name="mock-node", domain="text", **kwargs):
        super().__init__(name=name, domain=domain, **kwargs)
        self._run_calls = 0
        self._last_input: MosaicData | None = None

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data: MosaicData) -> MosaicData:
        self._run_calls += 1
        self._last_input = input_data
        content = input_data.get("content", "mock-output")
        return TextData(content=content, language="en")

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


@pytest.fixture
def MockNode():
    """返回 mock Node 类（非实例）。"""
    return _MockNodeImpl


@pytest.fixture
def mock_node():
    """返回一个已实例化的 mock 节点。"""
    return _MockNodeImpl()


# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_text_data():
    """返回包含测试文本的 MosaicData。"""
    return TextData(content="Hello, Mosaic!", language="en")


@pytest.fixture
def sample_messages():
    """返回一组测试对话历史。"""
    return [
        {"role": "user", "content": "你好，请介绍一下自己。"},
        {"role": "assistant", "content": "你好！我是 Mosaic AI 助手。"},
        {"role": "user", "content": "你能做什么？"},
    ]


@pytest.fixture
def sample_mosaic_data():
    """返回一个通用 MosaicData，key=value 形式。"""
    return MosaicData(
        prompt="测试 prompt",
        text="测试文本",
        max_new_tokens=128,
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# 调度器与事件总线
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_bus():
    """返回一个清理后的事件总线。"""
    EventBus._reset_singleton()
    bus = EventBus()
    bus.clear()
    return bus


@pytest.fixture
def cpu_scheduler():
    """返回一个 CPU 模式调度器。"""
    from mosaic.core.scheduler import set_scheduler

    bus = EventBus()
    bus.clear()
    sched = Scheduler(bus=bus, device="cpu")
    set_scheduler(sched)
    return sched


@pytest.fixture
def gpu_scheduler():
    """返回一个模拟 GPU 模式调度器（memory_limit=10GB）。"""
    from mosaic.core.scheduler import set_scheduler

    bus = EventBus()
    bus.clear()
    sched = Scheduler(bus=bus, device="cuda", memory_limit_gb=10.0)
    set_scheduler(sched)
    return sched


# ---------------------------------------------------------------------------
# 真实模型的文本节点（需要 transformers 时使用，默认 skip）
# ---------------------------------------------------------------------------
def _has_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


requires_transformers = pytest.mark.skipif(
    not _has_transformers(),
    reason="transformers not installed; skip real-model tests.",
)


# ---------------------------------------------------------------------------
# 工具辅助
# ---------------------------------------------------------------------------
@pytest.fixture
def clear_registry():
    """清空全局注册表。"""
    from mosaic.core.registry import NodeRegistry

    reg = NodeRegistry()
    # 用新的注册表替换全局引用中的内容
    registry._nodes.clear()
    registry._instances.clear()
    return registry