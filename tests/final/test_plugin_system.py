# tests/final/test_plugin_system.py
"""Mosaic 最终验收测试 —— 插件系统测试。

覆盖：
1. @node 装饰器注册自定义节点
2. 自定义节点在注册表列表中可见
3. 自定义节点可用于管道组合
4. 插件加载失败不影响其他插件
5. PluginManager.list_plugins 包含自定义插件
6. 插件节点的 describe 信息完整
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mosaic.core import Node, NodeSpec, Pipeline, registry
from mosaic.core.plugin import plugin_manager, node, PluginInfo
from mosaic.core.types import TextData


# ===========================================================================
# 辅助：自定义节点工厂
# ===========================================================================
def _make_custom_node_class(node_name="my_custom_node"):
    """创建一个自定义节点类（未注册）。

    每次调用返回新类，通过 node_name 参数区分不同测试的节点名，
    避免跨测试的注册冲突。
    """
    _name = node_name  # 闭包捕获，避免 class body 中 name 变量冲突

    class MyCustomNode(Node):
        name = _name
        domain = "custom"
        description = "A custom test node"
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
                name=_name,
                domain="custom",
                description="A custom test node",
                version="0.1.0",
                input_types=["text"],
                output_types=["text"],
            )

    return MyCustomNode


# ===========================================================================
# 注册 + 清理 fixture
# ===========================================================================
@pytest.fixture
def registered_custom_node():
    """注册一个自定义节点并在测试结束后清理注册表。

    每个测试使用唯一的节点名称，避免跨测试冲突。
    """
    node_name = "my_custom_node"
    MyCustomNode = _make_custom_node_class(node_name=node_name)

    node(
        domain="custom",
        name=node_name,
        version="0.1.0",
        description="A custom test node",
        author="test",
    )(MyCustomNode)

    yield node_name

    # 清理：从注册表注销自定义节点
    try:
        registry.unregister(node_name)
    except Exception:
        pass


# ===========================================================================
# T_PLUGIN_01: @node decorator registers custom node
# ===========================================================================
def test_decorator_registers_custom_node(registered_custom_node):
    """T_PLUGIN_01: @node 装饰器将自定义节点注册到注册表。

    验证：
    - 节点名称出现在 registry 中
    - 可以通过 registry.get() 获取实例
    """
    node_name = registered_custom_node

    # 验证节点出现在注册表中
    assert node_name in registry.list_names(), (
        f"{node_name} should appear in registry.list_names()"
    )

    # 验证可以通过 registry.get 获取
    instance = registry.get(node_name)
    assert instance is not None, (
        f"registry.get('{node_name}') should return a node instance"
    )
    assert instance.name == node_name, (
        f"Node instance name should be '{node_name}', got: {instance.name}"
    )
    assert instance.domain == "custom", (
        f"Node instance domain should be 'custom', got: {instance.domain}"
    )


# ===========================================================================
# T_PLUGIN_02: Custom node appears in mosaic list
# ===========================================================================
def test_custom_node_in_registry_list(registered_custom_node):
    """T_PLUGIN_02: 自定义节点出现在 registry.list_names() 中。

    验证通过 @node 装饰器注册后，节点名称在注册表列表中可见。
    """
    node_name = registered_custom_node

    names = registry.list_names()
    assert node_name in names, (
        f"'{node_name}' should be in registry.list_names(), got: {names}"
    )


# ===========================================================================
# T_PLUGIN_03: Custom node can be used in pipeline
# ===========================================================================
def test_custom_node_in_pipeline(registered_custom_node):
    """T_PLUGIN_03: 自定义节点可用于管道组合。

    验证：
    - 管道元素包含自定义节点
    - 管道结构校验通过
    - 管道执行不崩溃
    """
    node_name = registered_custom_node

    # 创建包含自定义节点的管道
    custom_node_instance = registry.get(node_name)
    pipe = Pipeline("custom-pipe", [custom_node_instance])

    assert len(pipe.elements) == 1, (
        "Pipeline should have exactly 1 element"
    )

    # 验证管道结构
    pipe.validate()  # 不应抛出异常

    dry_result = pipe.dry_run()
    assert dry_result.ok, (
        f"Dry run should pass, got issues: {dry_result.issues}"
    )

    # 尝试执行管道
    result = pipe.execute_result(TextData(content="hello"), fail_fast=False)
    assert result is not None, "Pipeline execution should return a result"
    assert result.pipeline_name == "custom-pipe", (
        f"Pipeline name should be 'custom-pipe', got: {result.pipeline_name}"
    )


# ===========================================================================
# T_PLUGIN_04: Plugin load failure doesn't affect other plugins
# ===========================================================================
def test_plugin_load_failure_isolation(registered_custom_node, plugin_manager):
    """T_PLUGIN_04: 插件加载失败不影响其他插件。

    验证：
    - 加载包含语法错误的插件文件不会导致崩溃
    - 其他已注册的插件仍然正常工作
    """
    node_name = registered_custom_node

    # 创建临时目录，包含一个语法错误的插件文件
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_plugin_path = os.path.join(tmpdir, "bad_plugin.py")
        with open(bad_plugin_path, "w") as f:
            f.write("# This file has a syntax error\n")
            f.write("def broken():\n")
            f.write("    this is not valid Python!!!\n")

        # 注册临时目录到 plugin_manager
        plugin_manager.register_plugin_dir(tmpdir)

        # 调用 load_plugins —— 不应崩溃
        try:
            plugin_manager.load_plugins()
        except Exception as exc:
            pytest.fail(
                f"plugin_manager.load_plugins() should not crash on bad plugin: {exc}"
            )

        # 验证其他插件仍然正常（自定义节点应该还在）
        assert node_name in registry.list_names(), (
            f"{node_name} should still be in registry after loading bad plugins"
        )

        # 验证可以通过 registry.get 正常获取
        instance = registry.get(node_name)
        assert instance is not None, (
            f"Should still be able to get {node_name} after bad plugin load"
        )


# ===========================================================================
# T_PLUGIN_05: PluginManager.list_plugins includes custom plugins
# ===========================================================================
def test_plugin_manager_lists_decorator_plugins(registered_custom_node, plugin_manager):
    """T_PLUGIN_05: PluginManager.list_plugins 包含自定义插件。

    验证通过 @node 装饰器注册后，plugin_manager.list_plugins()
    返回的列表中包含对应的插件，且 source 为 "decorator"。
    """
    node_name = registered_custom_node

    # 获取所有插件
    plugins = plugin_manager.list_plugins()
    assert len(plugins) >= 1, (
        f"plugin_manager.list_plugins() should return at least 1 plugin, got: {len(plugins)}"
    )

    # 查找 source 为 "decorator" 的插件
    decorator_plugins = plugin_manager.list_plugins(source="decorator")
    assert len(decorator_plugins) >= 1, (
        f"Should have at least 1 decorator plugin, got: {len(decorator_plugins)}"
    )

    # 验证自定义节点在装饰器插件列表中
    plugin_names = [p.name for p in decorator_plugins]
    assert node_name in plugin_names, (
        f"'{node_name}' should be in decorator plugins, got: {plugin_names}"
    )

    # 验证可以通过 get_plugin 获取
    info = plugin_manager.get_plugin(node_name)
    assert info is not None, (
        f"plugin_manager.get_plugin('{node_name}') should return a PluginInfo"
    )
    assert isinstance(info, PluginInfo), (
        f"get_plugin should return PluginInfo, got: {type(info)}"
    )
    assert info.source == "decorator", (
        f"Plugin source should be 'decorator', got: {info.source}"
    )
    assert info.name == node_name, (
        f"Plugin name should be '{node_name}', got: {info.name}"
    )
    assert info.domain == "custom", (
        f"Plugin domain should be 'custom', got: {info.domain}"
    )


# ===========================================================================
# T_PLUGIN_06: Plugin node's describe info is complete
# ===========================================================================
def test_plugin_node_describe_complete(registered_custom_node):
    """T_PLUGIN_06: 插件节点的 describe 信息完整。

    验证 NodeSpec 包含 name, domain, description, version,
    input_types, output_types 等所有字段。
    """
    node_name = registered_custom_node

    custom_node = registry.get(node_name)
    spec = custom_node.describe()

    # 验证 NodeSpec 基本字段
    assert isinstance(spec, NodeSpec), (
        f"describe() should return NodeSpec, got: {type(spec)}"
    )
    assert spec.name == node_name, (
        f"NodeSpec name should be '{node_name}', got: {spec.name}"
    )
    assert spec.domain == "custom", (
        f"NodeSpec domain should be 'custom', got: {spec.domain}"
    )
    assert spec.description == "A custom test node", (
        f"NodeSpec description mismatch, got: {spec.description}"
    )
    assert spec.version == "0.1.0", (
        f"NodeSpec version should be '0.1.0', got: {spec.version}"
    )

    # 验证 input_types / output_types
    assert spec.input_types == ["text"], (
        f"NodeSpec input_types should be ['text'], got: {spec.input_types}"
    )
    assert spec.output_types == ["text"], (
        f"NodeSpec output_types should be ['text'], got: {spec.output_types}"
    )

    # 验证 to_dict() 方法
    spec_dict = spec.to_dict()
    assert spec_dict["name"] == node_name
    assert spec_dict["domain"] == "custom"
    assert spec_dict["description"] == "A custom test node"
    assert spec_dict["version"] == "0.1.0"
    assert spec_dict["input_types"] == ["text"]
    assert spec_dict["output_types"] == ["text"]