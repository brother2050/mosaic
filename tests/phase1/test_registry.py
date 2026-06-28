# tests/phase1/test_registry.py
"""Phase 1 注册表测试。

覆盖 NodeRegistry 的注册、查询、列举、自动扫描与错误处理。
"""

from __future__ import annotations

import pytest

from mosaic.core.node import Node, NodeSpec
from mosaic.core.registry import NodeRegistry, registry, get_default_registry
from mosaic.core.types import MosaicData


# ===========================================================================
# 辅助：用于测试注册的节点类
# ===========================================================================
class _TestNodeA(Node):
    name = "test-node-a"
    domain = "text"
    description = "Test node A"
    version = "1.0.0"
    input_types = ["text"]
    output_types = ["text"]

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        return input_data

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _TestNodeB(Node):
    name = "test-node-b"
    domain = "image"
    description = "Test node B"
    version = "0.5.0"
    input_types = ["image"]
    output_types = ["image"]

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        return input_data

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


class _TestNodeC(Node):
    name = "test-node-c"
    domain = "text"
    description = "Test node C"
    version = "2.0.0"
    input_types = ["text"]
    output_types = ["text"]

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        return input_data

    def describe(self):
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
        )


# ===========================================================================
# T_REG_01: 注册一个节点类，然后按名称获取
# ===========================================================================
class TestRegistryRegisterAndGet:
    """注册表基本注册/获取测试。"""

    def test_register_class(self, clear_registry):
        """T_REG_01: 注册一个节点类，按名称获取实例。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        assert "test-node-a" in reg
        instance = reg.get("test-node-a")
        assert isinstance(instance, _TestNodeA)
        assert instance.name == "test-node-a"

    def test_get_by_class_name(self, clear_registry):
        """T_REG_01: 可按类名获取。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        instance = reg.get("_TestNodeA")
        assert isinstance(instance, _TestNodeA)

    def test_get_with_kwargs_creates_new_instance(self, clear_registry):
        """T_REG_01: 带 kwargs 的 get 创建新实例而非缓存。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        inst1 = reg.get("test-node-a")
        inst2 = reg.get("test-node-a", domain="custom-domain")
        assert inst1 is not inst2
        assert inst2.domain == "custom-domain"

    def test_get_unregistered_raises(self, clear_registry):
        """T_REG_01: 获取未注册节点抛出 KeyError。"""
        reg = clear_registry
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_get_class(self, clear_registry):
        """T_REG_01: get_class 返回类而非实例。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        cls = reg.get_class("test-node-a")
        assert cls is _TestNodeA


# ===========================================================================
# T_REG_02: @registry.register 装饰器正常工作
# ===========================================================================
class TestRegistryDecorator:
    """@registry.register 装饰器测试。"""

    def test_decorator_registers_class(self, clear_registry):
        """T_REG_02: @registry.register 装饰器自动注册。"""
        reg = clear_registry

        @reg.register
        class _DecoratedNode(Node):
            name = "decorated-node"
            domain = "test"
            description = "Decorated"
            version = "0.1.0"
            input_types = ["text"]
            output_types = ["text"]

            def load(self):
                self._loaded = True

            def unload(self):
                self._loaded = False

            def run(self, input_data):
                return input_data

            def describe(self):
                return NodeSpec(
                    name=self.name,
                    domain=self.domain,
                    description=self.description,
                    version=self.version,
                    input_types=list(self.input_types),
                    output_types=list(self.output_types),
                )

        assert "decorated-node" in reg
        inst = reg.get("decorated-node")
        assert inst.name == "decorated-node"

    def test_decorator_returns_class(self, clear_registry):
        """T_REG_02: 装饰器原样返回类。"""
        reg = clear_registry

        @reg.register
        class _RetNode(Node):
            name = "ret-node"
            domain = "test"
            description = "Ret"
            version = "0.1.0"
            input_types = ["text"]
            output_types = ["text"]

            def load(self):
                self._loaded = True

            def unload(self):
                self._loaded = False

            def run(self, input_data):
                return input_data

            def describe(self):
                return NodeSpec(
                    name=self.name,
                    domain=self.domain,
                    description=self.description,
                    version=self.version,
                    input_types=list(self.input_types),
                    output_types=list(self.output_types),
                )

        assert issubclass(_RetNode, Node)

    def test_registry_skips_abstract_classes(self, clear_registry):
        """T_REG_02: 抽象类不会被注册。"""
        import inspect

        reg = clear_registry

        class _AbstractNode(Node):
            name = "abstract-node"
            domain = "test"
            # 故意不实现 load/unload/run/describe，使其保持抽象

        assert inspect.isabstract(_AbstractNode)
        reg.register(_AbstractNode)
        # 抽象类不应进入注册表
        assert "abstract-node" not in reg


# ===========================================================================
# T_REG_03: list_nodes 返回所有已注册节点
# ===========================================================================
class TestRegistryListNodes:
    """注册表列举节点测试。"""

    def test_list_nodes_returns_all(self, clear_registry):
        """T_REG_03: list_nodes 返回所有已注册节点 spec。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        reg.register(_TestNodeB)
        specs = reg.list_nodes()
        assert len(specs) >= 2
        names = [s.name for s in specs]
        assert "test-node-a" in names
        assert "test-node-b" in names

    def test_list_nodes_deduplicates(self, clear_registry):
        """T_REG_03: list_nodes 去重（同一类因 name+类名 登记两次）。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        specs = reg.list_nodes()
        # 应该只有一个 _TestNodeA 的 spec
        name_a_count = sum(1 for s in specs if s.name == "test-node-a")
        assert name_a_count == 1


# ===========================================================================
# T_REG_04: list_nodes("text") 只返回文本域节点
# ===========================================================================
class TestRegistryListNodesByDomain:
    """注册表按域过滤列举测试。"""

    def test_list_nodes_text_domain(self, clear_registry):
        """T_REG_04: list_nodes("text") 只返回文本域节点。"""
        reg = clear_registry
        reg.register(_TestNodeA)  # text
        reg.register(_TestNodeB)  # image
        reg.register(_TestNodeC)  # text
        specs = reg.list_nodes("text")
        names = [s.name for s in specs]
        assert "test-node-a" in names
        assert "test-node-c" in names
        assert "test-node-b" not in names

    def test_list_nodes_image_domain(self, clear_registry):
        """T_REG_04: list_nodes("image") 只返回图像域节点。"""
        reg = clear_registry
        reg.register(_TestNodeA)  # text
        reg.register(_TestNodeB)  # image
        specs = reg.list_nodes("image")
        names = [s.name for s in specs]
        assert "test-node-b" in names
        assert "test-node-a" not in names

    def test_list_nodes_unknown_domain(self, clear_registry):
        """T_REG_04: 未知域返回空列表。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        specs = reg.list_nodes("audio")
        assert specs == []


# ===========================================================================
# T_REG_05: list_domains 返回所有已注册的域
# ===========================================================================
class TestRegistryListDomains:
    """注册表列举域测试。"""

    def test_list_domains(self, clear_registry):
        """T_REG_05: list_domains 返回所有已注册域（去重排序）。"""
        reg = clear_registry
        reg.register(_TestNodeA)  # text
        reg.register(_TestNodeB)  # image
        reg.register(_TestNodeC)  # text
        domains = reg.list_domains()
        assert "text" in domains
        assert "image" in domains
        assert domains == sorted(domains)

    def test_list_names(self, clear_registry):
        """T_REG_05: list_names 返回所有节点名。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        reg.register(_TestNodeB)
        names = reg.list_names()
        assert "test-node-a" in names
        assert "test-node-b" in names

    def test_len(self, clear_registry):
        """T_REG_05: __len__ 返回已注册节点数量（去重）。"""
        reg = clear_registry
        assert len(reg) == 0
        reg.register(_TestNodeA)
        assert len(reg) == 1
        reg.register(_TestNodeB)
        assert len(reg) == 2


# ===========================================================================
# T_REG_06: 获取不存在的节点名返回 None 或抛出友好错误
# ===========================================================================
class TestRegistryErrors:
    """注册表错误处理测试。"""

    def test_get_unregistered_raises_key_error(self, clear_registry):
        """T_REG_06: 获取不存在的节点抛出 KeyError。"""
        reg = clear_registry
        with pytest.raises(KeyError, match="not registered"):
            reg.get("ghost-node")

    def test_get_class_unregistered_raises_key_error(self, clear_registry):
        """T_REG_06: get_class 不存在的节点抛出 KeyError。"""
        reg = clear_registry
        with pytest.raises(KeyError, match="not registered"):
            reg.get_class("ghost-node")

    def test_contains_returns_false(self, clear_registry):
        """T_REG_06: __contains__ 对不存在的节点返回 False。"""
        reg = clear_registry
        assert "ghost-node" not in reg

    def test_register_non_node_raises(self, clear_registry):
        """T_REG_06: 注册非 Node 子类抛出 TypeError。"""
        reg = clear_registry

        class _NotNode:
            name = "not-node"

        with pytest.raises(TypeError, match="Expected a subclass of Node"):
            reg.register(_NotNode)  # type: ignore

    def test_unregister_nonexistent(self, clear_registry):
        """T_REG_06: 注销不存在的节点不报错。"""
        reg = clear_registry
        reg.unregister("ghost")  # 不应抛出异常


# ===========================================================================
# T_REG_07: 重复注册同名节点的行为（警告或覆盖）
# ===========================================================================
class TestRegistryDuplicate:
    """注册表重复注册测试。"""

    def test_duplicate_name_raises_error(self, clear_registry):
        """T_REG_07: 重复注册同名节点抛出 ValueError。"""
        reg = clear_registry
        reg.register(_TestNodeA)

        class _AnotherA(Node):
            name = "test-node-a"  # 同名
            domain = "text"
            description = "Another A"
            version = "0.1.0"
            input_types = ["text"]
            output_types = ["text"]

            def load(self):
                self._loaded = True

            def unload(self):
                self._loaded = False

            def run(self, input_data):
                return input_data

            def describe(self):
                return NodeSpec(
                    name=self.name,
                    domain=self.domain,
                    description=self.description,
                    version=self.version,
                    input_types=list(self.input_types),
                    output_types=list(self.output_types),
                )

        with pytest.raises(ValueError, match="already registered"):
            reg.register(_AnotherA)

    def test_re_register_same_class_no_error(self, clear_registry):
        """T_REG_07: 重复注册同一个类不报错。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        # 再次注册同一个类
        reg.register(_TestNodeA)  # 不应抛出异常
        assert "test-node-a" in reg

    def test_unregister_removes_node(self, clear_registry):
        """T_REG_07: unregister 移除节点。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        assert "test-node-a" in reg
        reg.unregister("test-node-a")
        assert "test-node-a" not in reg

    def test_unregister_also_removes_class_name_alias(self, clear_registry):
        """T_REG_07: unregister 同时移除类名别名。"""
        reg = clear_registry
        reg.register(_TestNodeA)
        # 类名别名也应存在
        assert "_TestNodeA" in reg
        reg.unregister("test-node-a")
        # 类名别名也应被移除
        assert "_TestNodeA" not in reg


# ===========================================================================
# 补充：全局注册表
# ===========================================================================
class TestGlobalRegistry:
    """全局注册表测试。"""

    def test_get_default_registry(self):
        """get_default_registry 返回全局单例。"""
        reg = get_default_registry()
        assert isinstance(reg, NodeRegistry)

    def test_global_registry_is_same_instance(self):
        """全局注册表是单例。"""
        reg1 = get_default_registry()
        reg2 = get_default_registry()
        assert reg1 is reg2