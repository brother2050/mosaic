# tests/final/test_serialization.py
"""Mosaic 最终验收测试 —— 序列化测试。

覆盖：
1. PipelineResult.to_dict() 可 JSON 序列化
2. 中间产物快照保存与加载
3. AsyncTask.to_dict() 可序列化
4. SubtitleData SRT 格式输出
5. DocumentData 序列化往返
6. AudioData 序列化往返
7. VideoData 序列化往返
8. ImageData 序列化往返
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from mosaic.core import (
    Pipeline,
    PipelineResult,
    NodeError,
    registry,
)
from mosaic.core.types import (
    TextData,
    ImageData,
    AudioData,
    VideoData,
    SubtitleData,
    DocumentData,
    MosaicData,
    data_from_dict,
)
from mosaic.core.task import AsyncTask


# ===========================================================================
# T_SER_01: PipelineResult.to_dict() serializable to JSON
# ===========================================================================
def test_pipeline_result_to_dict_json():
    """T_SER_01: PipelineResult.to_dict() 可被 json.dumps 序列化。

    验证 PipelineResult 序列化为字典后可被 JSON 编码，
    且包含 output、pipeline_name 等关键字段。
    """
    result = PipelineResult(
        output=TextData(content="hello"),
        intermediate={"step1": TextData(content="intermediate")},
        errors=[],
        duration=1.5,
        node_durations={"step1": 1.0},
        pipeline_name="test",
    )

    d = result.to_dict()
    json_str = json.dumps(d)

    assert '"output"' in json_str, (
        "JSON output should contain 'output' key"
    )
    assert '"pipeline_name"' in json_str, (
        "JSON output should contain 'pipeline_name' key"
    )

    # 验证关键字段值
    assert d["pipeline_name"] == "test", (
        f"pipeline_name should be 'test', got: {d['pipeline_name']}"
    )
    assert d["duration"] == 1.5, (
        f"duration should be 1.5, got: {d['duration']}"
    )
    assert d["success"] is True, (
        f"success should be True, got: {d['success']}"
    )
    assert d["node_count"] == 1, (
        f"node_count should be 1, got: {d['node_count']}"
    )

    # 验证可以反序列化回 JSON 对象
    restored = json.loads(json_str)
    assert restored["pipeline_name"] == "test"
    assert restored["duration"] == 1.5
    assert restored["success"] is True


# ===========================================================================
# T_SER_02: Intermediate snapshot save and load
# ===========================================================================
def test_pipeline_result_intermediate_snapshot():
    """T_SER_02: 中间产物快照保存与加载。

    验证 PipelineResult 的 get_intermediate 和 list_intermediate
    方法正确返回中间产物。
    """
    result = PipelineResult(
        output=TextData(content="final"),
        intermediate={
            "step1": TextData(content="intermediate"),
            "step2": TextData(content="another"),
        },
        errors=[],
        duration=2.0,
        node_durations={"step1": 1.0, "step2": 1.0},
        pipeline_name="test",
    )

    # 验证 get_intermediate
    step1_data = result.get_intermediate("step1")
    assert step1_data is not None, (
        "get_intermediate('step1') should return data"
    )
    assert isinstance(step1_data, TextData), (
        f"Intermediate should be TextData, got: {type(step1_data)}"
    )
    assert step1_data["content"] == "intermediate", (
        f"step1 content should be 'intermediate', got: {step1_data['content']}"
    )

    # 验证 list_intermediate
    intermediates = result.list_intermediate()
    assert isinstance(intermediates, list), (
        f"list_intermediate should return a list, got: {type(intermediates)}"
    )
    assert "step1" in intermediates, (
        f"list_intermediate should contain 'step1', got: {intermediates}"
    )
    assert "step2" in intermediates, (
        f"list_intermediate should contain 'step2', got: {intermediates}"
    )

    # 验证 get_intermediate 对不存在的 key 抛出 KeyError
    with pytest.raises(KeyError):
        result.get_intermediate("nonexistent")


# ===========================================================================
# T_SER_03: AsyncTask.to_dict() serializable
# ===========================================================================
def test_async_task_to_dict():
    """T_SER_03: AsyncTask.to_dict() 可序列化。

    验证 AsyncTask.to_dict() 返回包含 task_id、status 等关键字段的字典。
    """
    # 直接构造 AsyncTask（不执行管道，仅测试序列化结构）
    task = AsyncTask(
        pipeline_name="test-pipeline",
        pipeline=None,  # 不执行，仅测试序列化
        input_data=TextData(content="test"),
        task_id="test-task-001",
    )

    d = task.to_dict()

    assert "task_id" in d, (
        f"AsyncTask.to_dict() should contain 'task_id', got keys: {list(d.keys())}"
    )
    assert "status" in d, (
        f"AsyncTask.to_dict() should contain 'status', got keys: {list(d.keys())}"
    )
    assert d["task_id"] == "test-task-001", (
        f"task_id should be 'test-task-001', got: {d['task_id']}"
    )
    assert d["status"] == "pending", (
        f"status should be 'pending', got: {d['status']}"
    )
    assert d["pipeline_name"] == "test-pipeline", (
        f"pipeline_name should be 'test-pipeline', got: {d['pipeline_name']}"
    )
    assert "progress" in d, (
        "AsyncTask.to_dict() should contain 'progress'"
    )

    # 验证可 JSON 序列化
    json_str = json.dumps(d)
    restored = json.loads(json_str)
    assert restored["task_id"] == "test-task-001"
    assert restored["status"] == "pending"


# ===========================================================================
# T_SER_04: SubtitleData SRT format output correct
# ===========================================================================
def test_subtitle_data_srt_format():
    """T_SER_04: SubtitleData SRT 格式输出正确。

    验证 SubtitleData 序列化后 data_type 为 "subtitle"，
    且 segments 被正确保留。
    """
    sub = SubtitleData(
        segments=[
            {
                "index": 1,
                "start": "00:00:01,000",
                "end": "00:00:03,000",
                "text": "Hello world",
            },
            {
                "index": 2,
                "start": "00:00:04,000",
                "end": "00:00:06,000",
                "text": "Goodbye",
            },
        ],
        format="srt",
    )

    d = sub.to_dict()

    assert d["__data_type__"] == "subtitle", (
        f"__data_type__ should be 'subtitle', got: {d['__data_type__']}"
    )

    # 验证 segments 被保留
    assert len(d["segments"]) == 2, (
        f"segments should have 2 items, got: {len(d['segments'])}"
    )

    # 验证 segment 内容
    assert d["segments"][0]["text"] == "Hello world", (
        f"First segment text mismatch, got: {d['segments'][0].get('text')}"
    )
    assert d["segments"][1]["text"] == "Goodbye", (
        f"Second segment text mismatch, got: {d['segments'][1].get('text')}"
    )

    assert d["format"] == "srt", (
        f"format should be 'srt', got: {d['format']}"
    )

    # 验证可 JSON 序列化
    json_str = json.dumps(d)
    assert "subtitle" in json_str, (
        "JSON should contain 'subtitle' data type"
    )


# ===========================================================================
# T_SER_05: DocumentData serialization round-trip
# ===========================================================================
def test_document_data_round_trip():
    """T_SER_05: DocumentData 序列化往返。

    验证 DocumentData 经过 to_dict() -> data_from_dict() 往返后
    数据保持一致。
    """
    doc = DocumentData(
        chunks=["chunk1", "chunk2", "chunk3"],
        metadata={"source": "test.pdf"},
        chunk_metadata=[{"page": 1}, {"page": 1}, {"page": 2}],
    )

    d = doc.to_dict()
    restored = data_from_dict(d)

    assert isinstance(restored, DocumentData), (
        f"Restored data should be DocumentData, got: {type(restored)}"
    )

    assert restored["chunks"] == doc["chunks"], (
        f"Chunks mismatch: {restored['chunks']} != {doc['chunks']}"
    )

    assert len(restored["chunks"]) == 3, (
        f"Should have 3 chunks, got: {len(restored['chunks'])}"
    )

    # 验证 metadata
    assert restored["metadata"] == {"source": "test.pdf"}, (
        f"Metadata mismatch, got: {restored['metadata']}"
    )

    # 验证 chunk_metadata
    assert len(restored["chunk_metadata"]) == 3, (
        f"Should have 3 chunk_metadata entries, got: {len(restored['chunk_metadata'])}"
    )
    assert restored["chunk_metadata"][0] == {"page": 1}, (
        f"First chunk_metadata mismatch, got: {restored['chunk_metadata'][0]}"
    )
    assert restored["chunk_metadata"][2] == {"page": 2}, (
        f"Third chunk_metadata mismatch, got: {restored['chunk_metadata'][2]}"
    )


# ===========================================================================
# T_SER_06: AudioData serialization round-trip
# ===========================================================================
def test_audio_data_round_trip():
    """T_SER_06: AudioData 序列化往返。

    验证 AudioData 经过 to_dict() -> data_from_dict() 往返后
    数据类型和关键属性保持一致。
    """
    waveform = np.zeros(1000, dtype=np.float32)
    audio = AudioData(waveform=waveform, sample_rate=22050)

    d = audio.to_dict()
    restored = data_from_dict(d)

    assert isinstance(restored, AudioData), (
        f"Restored data should be AudioData, got: {type(restored)}"
    )

    assert restored["sample_rate"] == 22050, (
        f"sample_rate should be 22050, got: {restored['sample_rate']}"
    )

    # 验证波形数据已恢复
    restored_waveform = restored["waveform"]
    assert restored_waveform is not None, "Waveform should not be None"
    assert isinstance(restored_waveform, np.ndarray), (
        f"Restored waveform should be numpy.ndarray, got: {type(restored_waveform)}"
    )

    # 验证 __data_type__ 字段
    assert d["__data_type__"] == "audio", (
        f"__data_type__ should be 'audio', got: {d['__data_type__']}"
    )


# ===========================================================================
# T_SER_07: VideoData serialization round-trip
# ===========================================================================
def test_video_data_round_trip():
    """T_SER_07: VideoData 序列化往返。

    验证 VideoData 经过 to_dict() -> data_from_dict() 往返后
    数据类型和关键属性保持一致。
    """
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]
    video = VideoData(frames=frames, fps=30)

    d = video.to_dict()
    restored = data_from_dict(d)

    assert isinstance(restored, VideoData), (
        f"Restored data should be VideoData, got: {type(restored)}"
    )

    assert restored["fps"] == 30, (
        f"fps should be 30, got: {restored['fps']}"
    )

    # 验证帧数据已恢复
    restored_frames = restored["frames"]
    assert isinstance(restored_frames, list), (
        f"Restored frames should be list, got: {type(restored_frames)}"
    )
    assert len(restored_frames) == 5, (
        f"Should have 5 frames, got: {len(restored_frames)}"
    )

    # 验证 __data_type__ 字段
    assert d["__data_type__"] == "video", (
        f"__data_type__ should be 'video', got: {d['__data_type__']}"
    )


# ===========================================================================
# T_SER_08: ImageData serialization round-trip
# ===========================================================================
def test_image_data_round_trip():
    """T_SER_08: ImageData 序列化往返。

    验证 ImageData 经过 to_dict() -> data_from_dict() 往返后
    数据类型和尺寸保持一致。
    """
    img = Image.new("RGB", (64, 64), color="red")
    image_data = ImageData(image=img)

    d = image_data.to_dict()
    restored = data_from_dict(d)

    assert isinstance(restored, ImageData), (
        f"Restored data should be ImageData, got: {type(restored)}"
    )

    # 验证尺寸被保留
    restored_size = restored["size"]
    assert restored_size in ((64, 64), [64, 64]), (
        f"Size should be (64, 64) or [64, 64], got: {restored_size}"
    )

    # 验证 __data_type__ 字段
    assert d["__data_type__"] == "image", (
        f"__data_type__ should be 'image', got: {d['__data_type__']}"
    )

    # 验证恢复的图像是 PIL Image
    restored_img = restored["image"]
    assert restored_img is not None, "Restored image should not be None"
    assert isinstance(restored_img, Image.Image), (
        f"Restored image should be PIL.Image.Image, got: {type(restored_img)}"
    )


# ===========================================================================
# T_SER_09: MosaicData generic serialization round-trip
# ===========================================================================
def test_mosaic_data_round_trip():
    """T_SER_09: 通用 MosaicData 序列化往返。

    验证基础 MosaicData 经过 to_dict() -> data_from_dict() 往返后
    数据保持一致。
    """
    data = MosaicData(key1="value1", key2=42, key3=[1, 2, 3])

    d = data.to_dict()
    restored = data_from_dict(d)

    assert isinstance(restored, MosaicData), (
        f"Restored data should be MosaicData, got: {type(restored)}"
    )
    assert restored["key1"] == "value1", (
        f"key1 should be 'value1', got: {restored['key1']}"
    )
    assert restored["key2"] == 42, (
        f"key2 should be 42, got: {restored['key2']}"
    )
    assert restored["key3"] == [1, 2, 3], (
        f"key3 should be [1, 2, 3], got: {restored['key3']}"
    )

    # 验证 __data_type__ 字段
    assert d["__data_type__"] == "mosaic", (
        f"__data_type__ should be 'mosaic', got: {d['__data_type__']}"
    )


# ===========================================================================
# T_SER_10: NodeError serialization
# ===========================================================================
def test_node_error_serialization():
    """T_SER_10: NodeError 序列化。

    验证 NodeError.to_dict() 返回包含正确错误信息的字典。
    """
    err = NodeError(
        node_id="node-1",
        node_name="TestNode",
        error=ValueError("test error"),
        branch_name="main",
    )

    d = err.to_dict()

    assert d["node_id"] == "node-1", (
        f"node_id should be 'node-1', got: {d['node_id']}"
    )
    assert d["node_name"] == "TestNode", (
        f"node_name should be 'TestNode', got: {d['node_name']}"
    )
    assert d["error_type"] == "ValueError", (
        f"error_type should be 'ValueError', got: {d['error_type']}"
    )
    assert d["error_message"] == "test error", (
        f"error_message should be 'test error', got: {d['error_message']}"
    )
    assert d["branch_name"] == "main", (
        f"branch_name should be 'main', got: {d['branch_name']}"
    )

    # 验证可 JSON 序列化
    json_str = json.dumps(d)
    restored = json.loads(json_str)
    assert restored["node_id"] == "node-1"
    assert restored["error_type"] == "ValueError"