# tests/phase2/test_pipeline_parallel.py

import time

import pytest

from mosaic.core.branch import Branch, Merge
from mosaic.core.pipeline import Pipeline
from mosaic.core.result import NodeError, PipelineResult
from mosaic.core.types import MosaicData


class TestPipelineParallel:
    """Tests for parallel branch execution in Pipeline."""

    # T_PAR_01
    def test_parallel_branches_execute_concurrently(self, MockImageNode):
        """Branch with two mock nodes truly parallel.

        Both nodes have delay=0.1s.  If they execute serially the total time
        would be ~0.2s; parallel execution should stay well below that.
        """
        pipe = Pipeline(
            "parallel-test",
            [
                Branch(
                    a=MockImageNode(delay=0.1, tag="a"),
                    b=MockImageNode(delay=0.1, tag="b"),
                ),
                Merge(),
            ],
        )
        input_data = MosaicData(content="hello")
        t0 = time.perf_counter()
        result = pipe.execute_result(input_data)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.2, (
            f"Expected total time < 0.2s (parallel), got {elapsed:.3f}s"
        )
        assert result.success

    # T_PAR_02
    def test_merge_default_strategy_is_dict_with_branch_keys(self, MockImageNode):
        """Merge default strategy produces a dict keyed by branch names."""
        pipe = Pipeline(
            "merge-default",
            [
                Branch(
                    a=MockImageNode(tag="a"),
                    b=MockImageNode(tag="b"),
                ),
                Merge(),
            ],
        )
        result = pipe.execute_result(MosaicData(content="hello"))
        output = result.output

        assert "a" in output, f"Expected key 'a' in output, got {list(output.keys())}"
        assert "b" in output, f"Expected key 'b' in output, got {list(output.keys())}"
        assert output["a"]["tag"] == "a"
        assert output["b"]["tag"] == "b"

    # T_PAR_03
    def test_merge_keep_returns_only_specified_branch(self, MockImageNode):
        """Merge(keep="a") only keeps the specified branch result."""
        pipe = Pipeline(
            "merge-keep",
            [
                Branch(
                    a=MockImageNode(tag="a"),
                    b=MockImageNode(tag="b"),
                ),
                Merge(keep="a"),
            ],
        )
        result = pipe.execute_result(MosaicData(content="hello"))
        output = result.output

        assert output["tag"] == "a"
        assert "b" not in output

    # T_PAR_04
    def test_merge_custom_merge_function_is_called(self, MockImageNode):
        """Custom merge function is invoked and its return value is used."""
        pipe = Pipeline(
            "merge-custom",
            [
                Branch(
                    a=MockImageNode(tag="a"),
                    b=MockImageNode(tag="b"),
                ),
                Merge(merge_fn=lambda d: MosaicData(combined="merged")),
            ],
        )
        result = pipe.execute_result(MosaicData(content="hello"))
        output = result.output

        assert output["combined"] == "merged"

    # T_PAR_05
    def test_parallel_branch_input_each_branch_receives_input(self, MockImageNode):
        """Verify each branch receives the correct input in parallel execution."""
        pipe = Pipeline(
            "input-dist",
            [
                Branch(
                    a=MockImageNode(tag="a"),
                    b=MockImageNode(tag="b"),
                    input_strategy="copy",
                ),
                Merge(),
            ],
        )
        result = pipe.execute_result(MosaicData(content="hello"))
        output = result.output

        # Both branches received the same input and produced tagged output
        assert output["a"]["content"] == "hello->a"
        assert output["b"]["content"] == "hello->b"

    # T_PAR_06
    def test_fail_fast_true_stops_immediately_on_branch_failure(
        self, MockImageNode, FailingNode
    ):
        """fail_fast=True (default) -- one branch fails, execution stops immediately."""
        pipe = Pipeline(
            "fail-fast",
            [
                Branch(
                    a=MockImageNode(delay=0.1, tag="a"),
                    failing=FailingNode(error_msg="branch failed"),
                ),
                Merge(),
            ],
        )
        with pytest.raises(RuntimeError, match="branch failed"):
            pipe.execute_result(MosaicData(content="hello"))

    # T_PAR_07
    def test_fail_fast_false_other_branches_complete_errors_collected(
        self, MockImageNode, FailingNode
    ):
        """fail_fast=False -- failing branch is collected, other branches complete."""
        pipe = Pipeline(
            "fail-fast-false",
            [
                Branch(
                    a=MockImageNode(tag="a"),
                    failing=FailingNode(error_msg="branch failed"),
                ),
                Merge(),
            ],
        )
        result = pipe.execute_result(MosaicData(content="hello"), fail_fast=False)

        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0].node_name == "failing-node"
        assert "branch failed" in str(result.errors[0].error)
        assert isinstance(result.errors[0], NodeError)
        # The successful branch and the merge node should both have produced outputs
        assert len(result.intermediate) >= 1

    # T_PAR_08
    def test_three_way_parallel_all_branches_execute(self, MockImageNode):
        """3-way parallel -- Branch with 3 paths, verify all 3 execute."""
        pipe = Pipeline(
            "three-way",
            [
                Branch(
                    a=MockImageNode(tag="a"),
                    b=MockImageNode(tag="b"),
                    c=MockImageNode(tag="c"),
                ),
                Merge(),
            ],
        )
        result = pipe.execute_result(MosaicData(content="hello"))
        output = result.output

        assert "a" in output
        assert "b" in output
        assert "c" in output
        assert output["a"]["tag"] == "a"
        assert output["b"]["tag"] == "b"
        assert output["c"]["tag"] == "c"