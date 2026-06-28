# tests/phase2/test_intermediate.py
"""Phase 2 中间产物存储测试。

覆盖 Context 的中间产物存储、PipelineResult 的中间产物访问、
快照导出/导入以及 max_intermediate FIFO 淘汰机制。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from mosaic.core.context import Context, NodeOutput
from mosaic.core.node import Node, NodeSpec
from mosaic.core.pipeline import Pipeline, Branch, Merge
from mosaic.core.result import PipelineResult
from mosaic.core.types import MosaicData


# ===========================================================================
# 辅助：创建简单的 mock 节点类（用于 Pipeline 测试）
# ===========================================================================
class _EchoNode(Node):
    """简单的 echo 节点：将输入原样返回，并在数据上打标签。"""

    name = "echo"
    domain = "core"
    description = "Echo node for testing."
    version = "0.1.0"
    input_types = ["mosaic"]
    output_types = ["mosaic"]

    def __init__(self, name="echo", tag="", **kwargs):
        super().__init__(name=name, **kwargs)
        self._tag = tag

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def run(self, input_data):
        return MosaicData(
            content=f"{input_data.get('content', '')}->{self._tag}",
            tag=self._tag,
        )

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
# T_INT_01: execute_result() 后 get_intermediate() 可以取出节点输出
# ===========================================================================
class TestExecuteResultGetIntermediate:
    """测试通过 Pipeline.execute_result() 获取中间产物。"""

    def test_get_intermediate_retrieves_node_output(self):
        """T_INT_01: 运行 pipeline 后，result.get_intermediate() 可取出节点输出。"""
        node_a = _EchoNode(name="node_a", tag="A")
        node_b = _EchoNode(name="node_b", tag="B")
        pipe = Pipeline("test-pipe", [node_a, node_b])

        input_data = MosaicData(content="start")
        result = pipe.execute_result(input_data)

        # 验证 result.intermediate 字典非空
        assert len(result.intermediate) >= 2

        # 通过 get_intermediate() 按节点名获取
        # 注意：PipelineResult.get_intermediate() 先精确匹配 node_id，再按 name 模糊匹配
        output_a = result.get_intermediate("node_a")
        assert isinstance(output_a, MosaicData)
        assert "start->A" in output_a.get("content", "")

        output_b = result.get_intermediate("node_b")
        assert isinstance(output_b, MosaicData)
        assert "start->A->B" in output_b.get("content", "")

    def test_get_intermediate_raises_keyerror_for_missing(self):
        """T_INT_01: 获取不存在的节点产物应抛出 KeyError。"""
        node_a = _EchoNode(name="node_a", tag="A")
        pipe = Pipeline("test-pipe", [node_a])
        input_data = MosaicData(content="start")
        result = pipe.execute_result(input_data)

        with pytest.raises(KeyError, match="nonexistent"):
            result.get_intermediate("nonexistent")

    def test_intermediate_dict_contains_node_ids(self):
        """T_INT_01: result.intermediate 字典以 node_id 为键。"""
        node_a = _EchoNode(name="node_a", tag="A")
        node_b = _EchoNode(name="node_b", tag="B")
        pipe = Pipeline("test-pipe", [node_a, node_b])
        input_data = MosaicData(content="start")
        result = pipe.execute_result(input_data)

        # 中间产物字典的键是 node_id
        for key in result.intermediate:
            assert isinstance(key, str)
            assert isinstance(result.intermediate[key], MosaicData)


# ===========================================================================
# T_INT_02: list_intermediate() 返回所有中间节点名
# ===========================================================================
class TestExecuteResultListIntermediate:
    """测试 PipelineResult.list_intermediate()。"""

    def test_list_intermediate_returns_all_node_ids(self):
        """T_INT_02: list_intermediate() 返回所有中间产物节点 ID 列表。"""
        node_a = _EchoNode(name="node_a", tag="A")
        node_b = _EchoNode(name="node_b", tag="B")
        pipe = Pipeline("test-pipe", [node_a, node_b])
        input_data = MosaicData(content="start")
        result = pipe.execute_result(input_data)

        names = result.list_intermediate()
        assert isinstance(names, list)
        assert len(names) >= 2

        # 每个元素都是字符串（node_id）
        for name in names:
            assert isinstance(name, str)

    def test_list_intermediate_matches_intermediate_keys(self):
        """T_INT_02: list_intermediate() 返回的列表与 intermediate dict 键一致。"""
        node_a = _EchoNode(name="node_a", tag="A")
        pipe = Pipeline("test-pipe", [node_a])
        input_data = MosaicData(content="start")
        result = pipe.execute_result(input_data)

        names = result.list_intermediate()
        assert set(names) == set(result.intermediate.keys())

    def test_list_intermediate_after_multi_node_pipeline(self):
        """T_INT_02: 三节点管道 list_intermediate() 返回全部中间产物名。"""
        nodes = [
            _EchoNode(name="n1", tag="1"),
            _EchoNode(name="n2", tag="2"),
            _EchoNode(name="n3", tag="3"),
        ]
        pipe = Pipeline("multi-pipe", nodes)
        input_data = MosaicData(content="start")
        result = pipe.execute_result(input_data)

        names = result.list_intermediate()
        assert len(names) >= 3


# ===========================================================================
# T_INT_03: get_all_intermediate() 返回所有中间产物字典
# ===========================================================================
class TestGetAllIntermediate:
    """测试 Context.get_all_intermediate()。"""

    def test_get_all_intermediate_returns_dict(self):
        """T_INT_03: get_all_intermediate() 返回所有中间产物字典。"""
        ctx = Context()
        data_a = MosaicData(content="output_a")
        data_b = MosaicData(content="output_b")

        ctx.store_artifact("node_a", data_a)
        ctx.store_artifact("node_b", data_b)

        all_artifacts = ctx.get_all_intermediate()
        assert isinstance(all_artifacts, dict)
        assert len(all_artifacts) == 2
        assert "node_a" in all_artifacts
        assert "node_b" in all_artifacts
        assert all_artifacts["node_a"].get("content") == "output_a"
        assert all_artifacts["node_b"].get("content") == "output_b"

    def test_get_all_intermediate_empty(self):
        """T_INT_03: 无中间产物时返回空字典。"""
        ctx = Context()
        all_artifacts = ctx.get_all_intermediate()
        assert isinstance(all_artifacts, dict)
        assert len(all_artifacts) == 0

    def test_get_all_intermediate_after_pipeline(self):
        """T_INT_03: 管道执行后 Context.get_all_intermediate() 包含所有节点产物。"""
        node_a = _EchoNode(name="node_a", tag="A")
        node_b = _EchoNode(name="node_b", tag="B")
        pipe = Pipeline("test-pipe", [node_a, node_b])

        ctx = Context()
        input_data = MosaicData(content="start")
        result = pipe.execute_result(input_data, context=ctx)

        all_artifacts = ctx.get_all_intermediate()
        assert len(all_artifacts) >= 2


# ===========================================================================
# T_INT_04: Context.snapshot() 导出可序列化字典
# ===========================================================================
class TestSnapshot:
    """测试 Context.snapshot()。"""

    def test_snapshot_exports_serializable_dict(self):
        """T_INT_04: Context.snapshot() 导出可 JSON 序列化的字典。"""
        ctx = Context()
        ctx.store_artifact("node_a", MosaicData(content="hello"))
        ctx.store_artifact("node_b", MosaicData(content="world"))

        snap = ctx.snapshot()

        # 顶层结构
        assert isinstance(snap, dict)
        assert "config" in snap
        assert "artifacts" in snap
        assert "artifact_order" in snap

        # config 字段
        assert snap["config"]["device"] == "cpu"
        assert snap["config"]["precision"] == "fp32"

        # artifacts 字段
        assert "node_a" in snap["artifacts"]
        assert "node_b" in snap["artifacts"]
        assert "data" in snap["artifacts"]["node_a"]
        assert "timestamp" in snap["artifacts"]["node_a"]
        assert "duration" in snap["artifacts"]["node_a"]

        # artifact_order 字段
        assert snap["artifact_order"] == ["node_a", "node_b"]

        # 验证可 JSON 序列化
        json_str = json.dumps(snap, ensure_ascii=False)
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["config"]["device"] == "cpu"

    def test_snapshot_empty_context(self):
        """T_INT_04: 空上下文的 snapshot 只含 config 和空 artifacts。"""
        ctx = Context()
        snap = ctx.snapshot()
        assert snap["artifacts"] == {}
        assert snap["artifact_order"] == []

    def test_snapshot_includes_duration(self):
        """T_INT_04: snapshot 中的产物包含 duration 字段。"""
        ctx = Context()
        ctx.store_artifact("node_a", MosaicData(content="test"), duration=1.23)

        snap = ctx.snapshot()
        assert snap["artifacts"]["node_a"]["duration"] == 1.23


# ===========================================================================
# T_INT_05: Context.save_snapshot() 保存到文件，验证文件存在
# ===========================================================================
class TestSaveSnapshot:
    """测试 Context.save_snapshot()。"""

    def test_save_snapshot_creates_file(self):
        """T_INT_05: save_snapshot() 保存到文件，验证文件存在。"""
        ctx = Context()
        ctx.store_artifact("node_a", MosaicData(content="hello"))
        ctx.store_artifact("node_b", MosaicData(content="world"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            ctx.save_snapshot(tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0

            # 验证文件内容为有效 JSON
            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "config" in data
            assert "artifacts" in data
            assert "node_a" in data["artifacts"]
            assert "node_b" in data["artifacts"]
        finally:
            os.unlink(tmp_path)

    def test_save_snapshot_with_tmp_path(self, tmp_path):
        """T_INT_05: 使用 pytest tmp_path 保存快照文件。"""
        ctx = Context()
        ctx.store_artifact("node_x", MosaicData(content="data"))

        snap_path = tmp_path / "snapshot.json"
        ctx.save_snapshot(str(snap_path))

        assert snap_path.exists()
        assert snap_path.stat().st_size > 0

    def test_save_snapshot_overwrites_existing(self, tmp_path):
        """T_INT_05: save_snapshot() 覆盖已有文件。"""
        ctx = Context()
        ctx.store_artifact("node_a", MosaicData(content="first"))

        snap_path = tmp_path / "snapshot.json"
        ctx.save_snapshot(str(snap_path))
        size_first = snap_path.stat().st_size

        # 覆盖保存
        ctx2 = Context()
        ctx2.store_artifact("node_b", MosaicData(content="second"))
        ctx2.save_snapshot(str(snap_path))

        # 文件存在且内容已更新
        assert snap_path.exists()
        with open(str(snap_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "node_b" in data["artifacts"]
        assert "node_a" not in data["artifacts"]


# ===========================================================================
# T_INT_06: Context.load_snapshot() 从文件加载，验证产物恢复
# ===========================================================================
class TestLoadSnapshot:
    """测试 Context.load_snapshot()。"""

    def test_load_snapshot_restores_artifacts(self, tmp_path):
        """T_INT_06: load_snapshot() 从文件加载，验证产物恢复。"""
        # 先保存一个快照
        ctx_save = Context()
        ctx_save.store_artifact("node_a", MosaicData(content="hello"))
        ctx_save.store_artifact("node_b", MosaicData(content="world"))

        snap_path = tmp_path / "snapshot.json"
        ctx_save.save_snapshot(str(snap_path))

        # 加载到新 Context
        ctx_load = Context()
        ctx_load.load_snapshot(str(snap_path))

        # 验证产物已恢复
        assert ctx_load.has_artifact("node_a")
        assert ctx_load.has_artifact("node_b")
        assert ctx_load.get_artifact("node_a").get("content") == "hello"
        assert ctx_load.get_artifact("node_b").get("content") == "world"

    def test_load_snapshot_replaces_existing_artifacts(self, tmp_path):
        """T_INT_06: load_snapshot() 替换当前所有产物。"""
        # 目标 Context 已有产物
        ctx_load = Context()
        ctx_load.store_artifact("old_node", MosaicData(content="old"))

        # 保存快照（不同产物）
        ctx_save = Context()
        ctx_save.store_artifact("new_node", MosaicData(content="new"))
        snap_path = tmp_path / "snapshot.json"
        ctx_save.save_snapshot(str(snap_path))

        # 加载快照
        ctx_load.load_snapshot(str(snap_path))

        # 旧产物被清除，新产物存在
        assert not ctx_load.has_artifact("old_node")
        assert ctx_load.has_artifact("new_node")
        assert ctx_load.get_artifact("new_node").get("content") == "new"

    def test_load_snapshot_list_intermediate_after_load(self, tmp_path):
        """T_INT_06: 加载快照后 list_intermediate() 返回正确列表。"""
        ctx_save = Context()
        ctx_save.store_artifact("a", MosaicData(content="1"))
        ctx_save.store_artifact("b", MosaicData(content="2"))
        ctx_save.store_artifact("c", MosaicData(content="3"))

        snap_path = tmp_path / "snapshot.json"
        ctx_save.save_snapshot(str(snap_path))

        ctx_load = Context()
        ctx_load.load_snapshot(str(snap_path))

        names = ctx_load.list_intermediate()
        assert sorted(names) == ["a", "b", "c"]

    def test_load_snapshot_get_all_intermediate_after_load(self, tmp_path):
        """T_INT_06: 加载快照后 get_all_intermediate() 返回完整字典。"""
        ctx_save = Context()
        ctx_save.store_artifact("x", MosaicData(content="X"))
        ctx_save.store_artifact("y", MosaicData(content="Y"))

        snap_path = tmp_path / "snapshot.json"
        ctx_save.save_snapshot(str(snap_path))

        ctx_load = Context()
        ctx_load.load_snapshot(str(snap_path))

        all_artifacts = ctx_load.get_all_intermediate()
        assert len(all_artifacts) == 2
        assert all_artifacts["x"].get("content") == "X"
        assert all_artifacts["y"].get("content") == "Y"


# ===========================================================================
# T_INT_07: max_intermediate 限制存储数量，FIFO 淘汰
# ===========================================================================
class TestMaxIntermediate:
    """测试 max_intermediate FIFO 淘汰机制。"""

    def test_max_intermediate_limits_storage(self):
        """T_INT_07: max_intermediate=2，存储 3 个产物，验证只有 2 个保留（FIFO 淘汰）。"""
        ctx = Context(max_intermediate=2)

        ctx.store_artifact("node_a", MosaicData(content="first"))
        ctx.store_artifact("node_b", MosaicData(content="second"))
        ctx.store_artifact("node_c", MosaicData(content="third"))

        # 应该只有 2 个产物保留
        names = ctx.list_intermediate()
        assert len(names) == 2

        # 最早存入的 node_a 应被淘汰（FIFO）
        assert not ctx.has_artifact("node_a")

        # node_b 和 node_c 应保留
        assert ctx.has_artifact("node_b")
        assert ctx.has_artifact("node_c")
        assert ctx.get_artifact("node_b").get("content") == "second"
        assert ctx.get_artifact("node_c").get("content") == "third"

    def test_max_intermediate_exact_limit(self):
        """T_INT_07: 存储数量恰好等于 max_intermediate 时全部保留。"""
        ctx = Context(max_intermediate=2)

        ctx.store_artifact("node_a", MosaicData(content="first"))
        ctx.store_artifact("node_b", MosaicData(content="second"))

        names = ctx.list_intermediate()
        assert len(names) == 2
        assert ctx.has_artifact("node_a")
        assert ctx.has_artifact("node_b")

    def test_max_intermediate_update_existing(self):
        """T_INT_07: 更新已存在的产物不触发淘汰（不增加新条目）。"""
        ctx = Context(max_intermediate=2)

        ctx.store_artifact("node_a", MosaicData(content="v1"))
        ctx.store_artifact("node_b", MosaicData(content="v2"))
        # 更新 node_a 不增加新条目，不触发淘汰
        ctx.store_artifact("node_a", MosaicData(content="v1_updated"))

        names = ctx.list_intermediate()
        assert len(names) == 2
        assert ctx.has_artifact("node_a")
        assert ctx.has_artifact("node_b")
        assert ctx.get_artifact("node_a").get("content") == "v1_updated"

    def test_max_intermediate_no_limit(self):
        """T_INT_07: max_intermediate=None 时不限制存储数量。"""
        ctx = Context(max_intermediate=None)

        for i in range(10):
            ctx.store_artifact(f"node_{i}", MosaicData(content=f"data_{i}"))

        assert len(ctx.list_intermediate()) == 10

    def test_max_intermediate_get_all_after_eviction(self):
        """T_INT_07: 淘汰后 get_all_intermediate() 只返回保留的产物。"""
        ctx = Context(max_intermediate=1)

        ctx.store_artifact("node_a", MosaicData(content="a"))
        ctx.store_artifact("node_b", MosaicData(content="b"))

        all_artifacts = ctx.get_all_intermediate()
        assert len(all_artifacts) == 1
        assert "node_a" not in all_artifacts
        assert "node_b" in all_artifacts


# ===========================================================================
# T_INT_01~T_INT_03 补充: 通过 Context 直接测试
# ===========================================================================
class TestContextDirectAccess:
    """直接通过 Context 测试中间产物 API。"""

    def test_store_and_get_artifact(self):
        """通过 Context.store_artifact() 和 get_artifact() 存取中间产物。"""
        ctx = Context()
        data = MosaicData(content="test_output")
        ctx.store_artifact("my_node", data)

        retrieved = ctx.get_artifact("my_node")
        assert isinstance(retrieved, MosaicData)
        assert retrieved.get("content") == "test_output"

    def test_get_intermediate_is_alias(self):
        """get_intermediate() 是 get_artifact() 的别名。"""
        ctx = Context()
        data = MosaicData(content="alias_test")
        ctx.store_artifact("alias_node", data)

        assert ctx.get_intermediate("alias_node").get("content") == "alias_test"
        assert ctx.get_intermediate("alias_node").get("content") == ctx.get_artifact("alias_node").get("content")

    def test_has_artifact(self):
        """has_artifact() 正确判断产物是否存在。"""
        ctx = Context()
        assert not ctx.has_artifact("missing")

        ctx.store_artifact("present", MosaicData(content="x"))
        assert ctx.has_artifact("present")

    def test_get_artifact_record(self):
        """get_artifact_record() 返回包含 timestamp 和 duration 的完整记录。"""
        ctx = Context()
        data = MosaicData(content="record_test")
        ctx.store_artifact("rec_node", data, duration=3.14)

        record = ctx.get_artifact_record("rec_node")
        assert isinstance(record, NodeOutput)
        assert record.data.get("content") == "record_test"
        assert record.duration == 3.14
        assert isinstance(record.timestamp, float)

    def test_get_artifact_raises_keyerror(self):
        """获取不存在的产物抛出 KeyError。"""
        ctx = Context()
        with pytest.raises(KeyError, match="missing_node"):
            ctx.get_artifact("missing_node")

    def test_get_node_durations(self):
        """get_node_durations() 返回各节点耗时。"""
        ctx = Context()
        ctx.store_artifact("fast", MosaicData(content="f"), duration=0.1)
        ctx.store_artifact("slow", MosaicData(content="s"), duration=2.5)

        durations = ctx.get_node_durations()
        assert durations["fast"] == 0.1
        assert durations["slow"] == 2.5