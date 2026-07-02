# test_$node_name_snake.py
"""$node_name 节点的单元测试骨架。

由 mosaic CLI (create-node) 自动生成。
请根据节点实际逻辑补充测试用例。
"""

from __future__ import annotations

import pytest

from mosaic.core.types import MosaicData

from $node_name_snake import $node_name


class Test$node_name:
    """$node_name 测试套件。"""

    @pytest.fixture
    def node(self) -> "$node_name":
        """创建节点实例。"""
        return $node_name()

    def test_describe(self, node: "$node_name") -> None:
        """测试 describe() 返回正确的规格说明。"""
        spec = node.describe()
        assert spec.name == "$node_name_snake"
        assert spec.domain == "$domain"

    def test_load_unload(self, node: "$node_name") -> None:
        """测试加载与卸载。"""
        node.load()
        assert node.is_loaded() is True
        node.unload()
        assert node.is_loaded() is False

    def test_run_not_implemented(self, node: "$node_name") -> None:
        """测试 run() 在未实现时抛出 NotImplementedError。"""
        with pytest.raises(NotImplementedError):
            node.run(MosaicData())

    # TODO: 根据节点实际逻辑补充更多测试用例
