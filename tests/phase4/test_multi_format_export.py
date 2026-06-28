# tests/phase4/test_multi_format_export.py
"""Phase 4 MultiFormatExporter 节点测试。

测试 MultiFormatExporter 的视频、图像、音频、字幕多格式导出功能，
包括 outputs 格式校验、不支持格式警告、total_size 计算等。
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from PIL import Image

from mosaic.core.types import (
    AudioData,
    ImageData,
    MosaicData,
    SubtitleData,
    VideoData,
)
from mosaic.nodes.export.multi_format_exporter import MultiFormatExporter


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _make_sample_frames(count=5):
    """创建合成帧列表。"""
    return [Image.new("RGB", (64, 64), color=(100, 150, 200)) for _ in range(count)]


def _make_sample_audio():
    """创建合成音频数据。"""
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    waveform = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    return AudioData(waveform=waveform, sample_rate=sr)


def _make_sample_subtitle():
    """创建合成字幕数据。"""
    return SubtitleData(
        segments=[
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.0, "end": 4.0, "text": "Second line"},
        ],
        format="srt",
    )


def _mock_video_encoder_success(output_path):
    """创建 mock VideoEncoder 返回成功结果。"""
    mock_encoder = MagicMock()
    mock_encoder.run.return_value = MosaicData(output_path=output_path)
    return mock_encoder


# ---------------------------------------------------------------------------
# TestMultiFormatExporter
# ---------------------------------------------------------------------------
class TestMultiFormatExporter:
    """MultiFormatExporter 多格式导出功能测试。"""

    # T_MULTI_01
    def test_video_multi_format_export(self):
        """T_MULTI_01：视频多格式导出 (mp4 + gif)，验证 outputs 字典。"""
        exporter = MultiFormatExporter()
        exporter.load()

        frames = _make_sample_frames()
        video_data = VideoData(frames=frames, fps=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = os.path.join(tmpdir, "test.mp4")

            # Mock VideoEncoder（在 video_encoder 模块中定义，_export_video 内部导入）
            with patch(
                "mosaic.nodes.export.video_encoder.VideoEncoder"
            ) as MockVE:
                mock_enc = MagicMock()
                mock_enc.run.return_value = MosaicData(output_path=mp4_path)
                MockVE.return_value = mock_enc

                # 预创建输出文件（使 size 计算可用）
                with open(mp4_path, "wb") as f:
                    f.write(b"mock-mp4-data-" * 20)

                result = exporter.run(
                    MosaicData(
                        content_type="video",
                        data=video_data,
                        formats=["mp4", "gif"],
                        output_dir=tmpdir,
                    )
                )

            outputs = result["outputs"]
            assert isinstance(outputs, dict), "outputs 应为 dict"
            assert "mp4" in outputs, "outputs 应包含 mp4"
            assert "gif" in outputs, "outputs 应包含 gif"
            assert outputs["mp4"] == mp4_path, f"mp4 路径应为 {mp4_path}"
            assert os.path.exists(outputs["gif"]), "gif 文件应存在"

            # 验证 mock 被调用
            assert MockVE.called, "VideoEncoder 应被调用用于 mp4 导出"

        exporter.unload()

    # T_MULTI_02
    def test_image_multi_format_export(self):
        """T_MULTI_02：图像多格式导出 (png + jpg + webp)，验证文件创建。"""
        exporter = MultiFormatExporter()
        exporter.load()

        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        image_data = ImageData(image=img)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = exporter.run(
                MosaicData(
                    content_type="image",
                    data=image_data,
                    formats=["png", "jpg", "webp"],
                    output_dir=tmpdir,
                    quality=50,
                )
            )

            outputs = result["outputs"]
            assert "png" in outputs, "outputs 应包含 png"
            assert "jpg" in outputs, "outputs 应包含 jpg"
            assert "webp" in outputs, "outputs 应包含 webp"

            for fmt in ("png", "jpg", "webp"):
                path = outputs[fmt]
                assert os.path.exists(path), f"[{fmt}] 文件应存在: {path}"
                assert os.path.getsize(path) > 0, f"[{fmt}] 文件不应为空: {path}"

        exporter.unload()

    # T_MULTI_03
    def test_audio_multi_format_export(self):
        """T_MULTI_03：音频多格式导出 (wav + mp3)，mock soundfile.write。"""
        exporter = MultiFormatExporter()
        exporter.load()

        audio = _make_sample_audio()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock soundfile.write 以创建实际文件
            def _mock_sf_write(path, data, sr, **kwargs):
                with open(path, "wb") as f:
                    f.write(b"mock-audio-data-" * 10)

            # 确保 soundfile mock 模块存在于 sys.modules 中
            import types as _types
            if "soundfile" not in sys.modules:
                _sf = _types.ModuleType("soundfile")
                _sf.write = MagicMock()
                sys.modules["soundfile"] = _sf

            with patch.object(sys.modules["soundfile"], "write", side_effect=_mock_sf_write):
                result = exporter.run(
                    MosaicData(
                        content_type="audio",
                        data=audio,
                        formats=["wav", "mp3"],
                        output_dir=tmpdir,
                        quality=50,
                    )
                )

            outputs = result["outputs"]
            assert "wav" in outputs, "outputs 应包含 wav"
            assert "mp3" in outputs, "outputs 应包含 mp3"

            for fmt in ("wav", "mp3"):
                path = outputs[fmt]
                assert os.path.exists(path), f"[{fmt}] 文件应存在: {path}"
                assert os.path.getsize(path) > 0, f"[{fmt}] 文件不应为空: {path}"

        exporter.unload()

    # T_MULTI_04
    def test_subtitle_multi_format_export(self):
        """T_MULTI_04：字幕多格式导出 (srt + vtt)，验证内容。"""
        exporter = MultiFormatExporter()
        exporter.load()

        subtitle = _make_sample_subtitle()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = exporter.run(
                MosaicData(
                    content_type="subtitle",
                    data=subtitle,
                    formats=["srt", "vtt"],
                    output_dir=tmpdir,
                )
            )

            outputs = result["outputs"]
            assert "srt" in outputs, "outputs 应包含 srt"
            assert "vtt" in outputs, "outputs 应包含 vtt"

            # 验证 SRT 内容
            srt_path = outputs["srt"]
            assert os.path.exists(srt_path), "SRT 文件应存在"
            with open(srt_path, "r", encoding="utf-8") as f:
                srt_content = f.read()
            assert "Hello world" in srt_content, "SRT 应包含原文"
            assert "00:00:00" in srt_content, "SRT 应包含时间戳"
            assert "Second line" in srt_content, "SRT 应包含第二行"

            # 验证 VTT 内容
            vtt_path = outputs["vtt"]
            assert os.path.exists(vtt_path), "VTT 文件应存在"
            with open(vtt_path, "r", encoding="utf-8") as f:
                vtt_content = f.read()
            assert "WEBVTT" in vtt_content, "VTT 应以 WEBVTT 开头"
            assert "Hello world" in vtt_content, "VTT 应包含原文"

        exporter.unload()

    # T_MULTI_05
    def test_outputs_dict_format_correct(self):
        """T_MULTI_05：outputs 字典格式正确 (format -> filepath 映射)。"""
        exporter = MultiFormatExporter()
        exporter.load()

        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        image_data = ImageData(image=img)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = exporter.run(
                MosaicData(
                    content_type="image",
                    data=image_data,
                    formats=["png", "jpg"],
                    output_dir=tmpdir,
                )
            )

            outputs = result["outputs"]
            assert isinstance(outputs, dict), "outputs 应为 dict"
            assert len(outputs) == 2, "outputs 应有 2 个条目"

            for fmt, filepath in outputs.items():
                assert isinstance(fmt, str), f"key 应为 str，实际 {type(fmt)}"
                assert isinstance(filepath, str), f"value 应为 str，实际 {type(filepath)}"
                assert os.path.isabs(filepath), f"文件路径应为绝对路径: {filepath}"
                assert filepath.endswith(f".{fmt}"), (
                    f"文件路径应以 .{fmt} 结尾: {filepath}"
                )

        exporter.unload()

    # T_MULTI_06
    def test_unsupported_format_warning(self, caplog):
        """T_MULTI_06：不支持的格式给出警告，被跳过。"""
        exporter = MultiFormatExporter()
        exporter.load()

        img = Image.new("RGB", (32, 32), color=(100, 200, 100))
        image_data = ImageData(image=img)

        with tempfile.TemporaryDirectory() as tmpdir, caplog.at_level(logging.WARNING):
            result = exporter.run(
                MosaicData(
                    content_type="image",
                    data=image_data,
                    formats=["png", "unsupported_fmt", "jpg"],
                    output_dir=tmpdir,
                )
            )

            outputs = result["outputs"]
            # 不支持格式应被跳过
            assert "unsupported_fmt" not in outputs, "不支持的格式不应出现在 outputs 中"
            assert "png" in outputs, "png 应正常导出"
            assert "jpg" in outputs, "jpg 应正常导出"

            # 检查警告日志
            warning_messages = [
                r.message for r in caplog.records if r.levelno >= logging.WARNING
            ]
            has_skip_warning = any(
                "Unsupported" in msg or "unsupported" in msg or "skipping" in msg.lower()
                for msg in warning_messages
            )
            assert has_skip_warning, (
                f"应发出不支持格式的警告日志，实际警告: {warning_messages}"
            )

        exporter.unload()

    # T_MULTI_07
    def test_total_size_calculated_correctly(self):
        """T_MULTI_07：total_size 计算正确。"""
        exporter = MultiFormatExporter()
        exporter.load()

        img = Image.new("RGB", (64, 64), color=(50, 100, 200))
        image_data = ImageData(image=img)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = exporter.run(
                MosaicData(
                    content_type="image",
                    data=image_data,
                    formats=["png", "jpg", "webp"],
                    output_dir=tmpdir,
                )
            )

            outputs = result["outputs"]
            # 手动计算各文件大小之和
            expected_total = sum(
                os.path.getsize(path) for path in outputs.values()
            )

            assert result["total_size"] == expected_total, (
                f"total_size 应为 {expected_total}，实际 {result['total_size']}"
            )
            assert result["total_size"] > 0, "total_size 应大于 0"
            assert isinstance(result["total_size"], int), "total_size 应为 int"

            # total_files
            assert result["total_files"] == len(outputs), (
                f"total_files 应为 {len(outputs)}，实际 {result['total_files']}"
            )

        exporter.unload()


class TestMultiFormatExporterEdgeCases:
    """MultiFormatExporter 边界情况测试。"""

    def test_missing_content_type_raises(self):
        """缺少 content_type 抛出 ValueError。"""
        exporter = MultiFormatExporter()
        exporter.load()

        with pytest.raises(ValueError, match="content_type"):
            exporter.run(MosaicData(data="test", formats=["mp4"]))

        exporter.unload()

    def test_missing_data_raises(self):
        """缺少 data 抛出 ValueError。"""
        exporter = MultiFormatExporter()
        exporter.load()

        with pytest.raises(ValueError, match="data"):
            exporter.run(MosaicData(content_type="video", formats=["mp4"]))

        exporter.unload()

    def test_missing_formats_raises(self):
        """缺少 formats 抛出 ValueError。"""
        exporter = MultiFormatExporter()
        exporter.load()

        img = Image.new("RGB", (32, 32))
        with pytest.raises(ValueError, match="formats"):
            exporter.run(
                MosaicData(content_type="image", data=ImageData(image=img))
            )

        exporter.unload()

    def test_empty_formats_raises(self):
        """空 formats 列表抛出 ValueError。"""
        exporter = MultiFormatExporter()
        exporter.load()

        img = Image.new("RGB", (32, 32))
        with pytest.raises(ValueError, match="formats"):
            exporter.run(
                MosaicData(
                    content_type="image",
                    data=ImageData(image=img),
                    formats=[],
                )
            )

        exporter.unload()

    def test_unsupported_content_type_raises(self):
        """不支持的内容类型抛出 ValueError。"""
        exporter = MultiFormatExporter()
        exporter.load()

        with pytest.raises(ValueError, match="Unsupported content_type"):
            exporter.run(
                MosaicData(
                    content_type="3d_model",
                    data="test",
                    formats=["obj"],
                )
            )

        exporter.unload()

    def test_describe_returns_model_info_none(self):
        """describe() 返回 NodeSpec，model_info=None。"""
        exporter = MultiFormatExporter()
        spec = exporter.describe()
        assert spec.name == "multi-format-exporter"
        assert spec.domain == "export"
        assert spec.model_info is None, "导出域节点 model_info 应为 None"