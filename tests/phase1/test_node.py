# tests/phase1/test_node.py
"""Phase 1 节点基类测试。

覆盖 Node 抽象基类的实例化、__call__、上下文管理器、加载状态与 describe。
"""

from __future__ import annotations

import pytest

from mosaic.core.node import Node, NodeSpec
from mosaic.core.types import MosaicData, TextData


# ===========================================================================
# 辅助：一个不完整的 Node 子类（用于测试抽象类行为）
# ===========================================================================
class _IncompleteNode(Node):
    name = "incomplete"
    domain = "test"


# ===========================================================================
# T_NODE_01: Node 是抽象类，不能直接实例化
# ===========================================================================
class TestNodeAbstract:
    """Node 抽象类测试。"""

    def test_cannot_instantiate_abstract_node(self):
        """T_NODE_01: 直接实例化 Node 抛出 TypeError。"""
        with pytest.raises(TypeError):
            Node()  # type: ignore

    def test_cannot_instantiate_incomplete_subclass(self):
        """T_NODE_01: 未实现所有抽象方法的子类也不能实例化。"""
        with pytest.raises(TypeError):
            _IncompleteNode()  # type: ignore

    def test_abstract_methods_are_required(self):
        """T_NODE_01: 抽象方法 load/unload/run/describe 必须实现。"""
        # 检查 _IncompleteNode 的抽象方法列表
        import inspect

        assert inspect.isabstract(_IncompleteNode)


# ===========================================================================
# T_NODE_02: 实现所有抽象方法后可以正常实例化
# ===========================================================================
class TestNodeInstantiation:
    """Node 实例化测试。"""

    def test_mock_node_can_be_instantiated(self, mock_node):
        """T_NODE_02: mock 节点可以正常实例化。"""
        assert mock_node.name == "mock-node"
        assert mock_node.domain == "text"
        assert not mock_node.is_loaded()

    def test_mock_node_class(self, MockNode):
        """T_NODE_02: mock 节点类可以实例化。"""
        node = MockNode(name="custom-name")
        assert node.name == "custom-name"
        assert node.domain == "text"

    def test_load_and_unload(self, mock_node):
        """T_NODE_02: load/unload 改变加载状态。"""
        assert not mock_node.is_loaded()
        mock_node.load()
        assert mock_node.is_loaded()
        mock_node.unload()
        assert not mock_node.is_loaded()

    def test_run_returns_data(self, mock_node, sample_text_data):
        """T_NODE_02: run 返回 MosaicData。"""
        mock_node.load()
        result = mock_node.run(sample_text_data)
        assert isinstance(result, TextData)
        assert result.content == "Hello, Mosaic!"

    def test_describe_returns_node_spec(self, mock_node):
        """T_NODE_02: describe 返回 NodeSpec。"""
        spec = mock_node.describe()
        assert isinstance(spec, NodeSpec)
        assert spec.name == "mock-node"
        assert spec.domain == "text"
        assert spec.input_types == ["text"]
        assert spec.output_types == ["text"]

    def test_accepts_and_produces(self, mock_node):
        """T_NODE_02: accepts / produces 方法。"""
        assert mock_node.accepts("text")
        assert not mock_node.accepts("image")
        assert mock_node.produces() == ["text"]


# ===========================================================================
# T_NODE_03: __call__ 方法正确调用 run
# ===========================================================================
class TestNodeCall:
    """Node.__call__ 测试。"""

    def test_call_invokes_load_and_run(self, mock_node, sample_text_data):
        """T_NODE_03: __call__ 自动 load 并调用 run。"""
        assert not mock_node.is_loaded()
        result = mock_node(sample_text_data)
        assert mock_node.is_loaded()
        assert isinstance(result, TextData)

    def test_call_tracks_run_count(self, mock_node, sample_text_data):
        """T_NODE_03: 每次 __call__ 增加 run 计数。"""
        assert mock_node._run_calls == 0
        mock_node(sample_text_data)
        assert mock_node._run_calls == 1
        mock_node(sample_text_data)
        assert mock_node._run_calls == 2

    def test_call_on_already_loaded(self, mock_node, sample_text_data):
        """T_NODE_03: 已加载节点 __call__ 不重复 load。"""
        mock_node.load()
        assert mock_node.is_loaded()
        mock_node(sample_text_data)
        assert mock_node.is_loaded()


# ===========================================================================
# T_NODE_04: 上下文管理器 __enter__ 调用 load，__exit__ 调用 unload
# ===========================================================================
class TestNodeContextManager:
    """Node 上下文管理器测试。"""

    def test_context_loads_and_unloads(self, mock_node):
        """T_NODE_04: 上下文管理器自动 load/unload。"""
        assert not mock_node.is_loaded()
        with mock_node as node:
            assert node.is_loaded()
            assert node is mock_node
        assert not mock_node.is_loaded()

    def test_context_exception_does_not_suppress(self, mock_node, sample_text_data):
        """T_NODE_04: 上下文内异常不吞掉。"""
        with pytest.raises(ValueError, match="test-exception"):
            with mock_node:
                raise ValueError("test-exception")
        # 即使异常，unload 也应被调用
        assert not mock_node.is_loaded()

    def test_nested_context(self, mock_node):
        """T_NODE_04: 嵌套上下文管理器。"""
        with mock_node:
            assert mock_node.is_loaded()
            with mock_node:
                assert mock_node.is_loaded()
            # 内层 __exit__ 调用 unload，节点已卸载
            assert not mock_node.is_loaded()
        # 外层 __exit__ 再次调用 unload（幂等）
        assert not mock_node.is_loaded()


# ===========================================================================
# T_NODE_05: is_loaded 状态正确反映
# ===========================================================================
class TestNodeLoadedState:
    """Node is_loaded 状态测试。"""

    def test_initial_state_is_unloaded(self, mock_node):
        """T_NODE_05: 初始状态为未加载。"""
        assert not mock_node.is_loaded()

    def test_state_after_load(self, mock_node):
        """T_NODE_05: load 后为已加载。"""
        mock_node.load()
        assert mock_node.is_loaded()

    def test_state_after_unload(self, mock_node):
        """T_NODE_05: unload 后为未加载。"""
        mock_node.load()
        assert mock_node.is_loaded()
        mock_node.unload()
        assert not mock_node.is_loaded()

    def test_state_after_double_unload(self, mock_node):
        """T_NODE_05: 重复 unload 不报错。"""
        mock_node.unload()
        assert not mock_node.is_loaded()
        mock_node.unload()
        assert not mock_node.is_loaded()


# ===========================================================================
# T_NODE_06: describe 返回正确的 NodeSpec
# ===========================================================================
class TestNodeDescribe:
    """Node.describe 测试。"""

    def test_node_spec_has_all_fields(self, mock_node):
        """T_NODE_06: NodeSpec 包含所有必要字段。"""
        spec = mock_node.describe()
        assert spec.name == "mock-node"
        assert spec.domain == "text"
        assert spec.description == "A mock node for testing."
        assert spec.version == "0.1.0"
        assert spec.input_types == ["text"]
        assert spec.output_types == ["text"]

    def test_node_spec_to_dict(self, mock_node):
        """T_NODE_06: NodeSpec.to_dict() 返回字典。"""
        spec = mock_node.describe()
        d = spec.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "mock-node"
        assert d["domain"] == "text"

    def test_node_spec_repr(self, mock_node):
        """T_NODE_06: NodeSpec.__repr__ 包含关键信息。"""
        spec = mock_node.describe()
        r = repr(spec)
        assert "NodeSpec" in r
        assert "mock-node" in r

    def test_node_repr(self, mock_node):
        """T_NODE_06: Node.__repr__ 包含状态信息。"""
        r = repr(mock_node)
        assert "unloaded" in r
        mock_node.load()
        r2 = repr(mock_node)
        assert "loaded" in r2


# ===========================================================================
# T_NODE_03 补充: 管道运算符 __or__
# ===========================================================================
class TestNodeOrOperator:
    """Node.__or__ 管道运算符测试。"""

    def test_node_or_node_creates_pipeline(self, mock_node, MockNode):
        """T_NODE_03: node_a | node_b 创建匿名 Pipeline。"""
        from mosaic.core.pipeline import Pipeline

        node_b = MockNode(name="mock-node-b")
        pipe = mock_node | node_b
        assert isinstance(pipe, Pipeline)
        assert pipe.name == "anonymous"
        assert len(pipe) == 2

    def test_node_or_non_node_raises_type_error(self, mock_node):
        """T_NODE_03: 非 Node 右操作数抛出 TypeError。"""
        with pytest.raises(TypeError, match="unsupported operand"):
            mock_node | "not a node"

    def test_node_or_pipeline(self, mock_node, MockNode):
        """T_NODE_03: Node | Pipeline 扁平展开。"""
        from mosaic.core.pipeline import Pipeline

        node_b = MockNode(name="b")
        node_c = MockNode(name="c")
        pipe = Pipeline("sub", [node_c])
        combined = mock_node | node_b | pipe
        assert isinstance(combined, Pipeline)
        # 扁平展开：node → node_b → node_c
        assert len(combined) == 3