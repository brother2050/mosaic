# mosaic/nodes/subtitle/_base.py
"""字幕域节点基类。

提取字幕生成/翻译/对齐节点共用的格式转换与片段处理逻辑。
子类只需实现 :meth:`BaseSubtitleNode.run` 与 :meth:`_load_model`，
字幕格式工具由本基类提供。

设计要点
--------
* SRT / WebVTT 格式的解析与生成完全自包含，不依赖外部库。
* 时间戳精度到毫秒级，符合 SRT/VTT 规范。
* 支持片段合并（过短）与拆分（过长），保证字幕可读性。
* 模型生命周期通过 :class:`~mosaic.core.scheduler.Scheduler` 管理。
* 关键步骤通过 :class:`~mosaic.core.events.EventBus` 发出事件。
"""

from __future__ import annotations

import abc
import logging
import re
from typing import Any

from mosaic.core.events import EventBus, EventType, get_event_bus
from mosaic.core.node import Node, NodeSpec
from mosaic.core.scheduler import Scheduler, get_scheduler
from mosaic.core.types import MosaicData, SubtitleData

__all__ = ["BaseSubtitleNode"]


class BaseSubtitleNode(Node):
    """字幕域节点抽象基类。

    封装字幕格式转换、片段处理与事件发射逻辑。子类需实现
    :meth:`run` 与 :meth:`_load_model`。

    Parameters
    ----------
    scheduler:
        显存调度器实例，``None`` 使用全局单例。
    bus:
        事件总线实例，``None`` 使用全局单例。
    """

    domain: str = "subtitle"
    description: str = "Base subtitle node."
    version: str = "0.1.0"
    input_types: list[str] = ["audio", "subtitle", "text", "mosaic"]
    output_types: list[str] = ["subtitle"]

    def __init__(
        self,
        scheduler: Scheduler | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._scheduler: Scheduler = scheduler or get_scheduler()
        self._bus: EventBus = bus or get_event_bus()
        self._logger = logging.getLogger(f"mosaic.nodes.subtitle.{self.name}")
        self._model: Any = None

    # ------------------------------------------------------------------
    # 模型加载 / 卸载（子类按需覆写）
    # ------------------------------------------------------------------
    def load(self) -> None:
        """加载模型（如有）。"""
        self._scheduler.track(self)
        if self._model is not None:
            self._loaded = True
            return
        self._load_model()
        self._loaded = True

    def _load_model(self) -> None:
        """子类实现：实际加载模型。默认无操作。"""
        pass

    def unload(self) -> None:
        """释放模型。"""
        self._model = None
        self._loaded = False

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
    # 字幕格式转换工具
    # ------------------------------------------------------------------
    @staticmethod
    def _format_timestamp(seconds: float, fmt: str = "srt") -> str:
        """将秒数格式化为时间戳字符串。

        Parameters
        ----------
        seconds:
            时间（秒）。
        fmt:
            格式类型：``"srt"`` → ``HH:MM:SS,mmm``，
            ``"vtt"`` → ``HH:MM:SS.mmm``。

        Returns
        -------
        str
            格式化后的时间戳。
        """
        if seconds < 0:
            seconds = 0.0

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds * 1000) % 1000)

        separator = "," if fmt == "srt" else "."
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"

    @staticmethod
    def _parse_timestamp(timestamp: str) -> float:
        """将时间戳字符串解析为秒数。

        支持 SRT (``HH:MM:SS,mmm``) 和 VTT (``HH:MM:SS.mmm``) 格式，
        也支持 ``MM:SS.mmm`` 等简短格式。

        Parameters
        ----------
        timestamp:
            时间戳字符串。

        Returns
        -------
        float
            秒数。
        """
        ts = timestamp.strip()
        # 统一逗号为点
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        try:
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            elif len(parts) == 1:
                return float(parts[0])
        except ValueError:
            pass
        return 0.0

    @staticmethod
    def _parse_srt(content: str) -> list[dict[str, Any]]:
        """解析 SRT 格式字幕内容。

        Parameters
        ----------
        content:
            SRT 格式字符串。

        Returns
        -------
        list[dict[str, Any]]
            字幕片段列表，每段含 ``index``/``start``/``end``/``text``。
        """
        segments: list[dict[str, Any]] = []
        # 按空行分割块
        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            if len(lines) < 2:
                continue

            idx = 0
            # 第一行可能是序号
            if lines[0].strip().isdigit():
                idx = int(lines[0].strip())
                time_line = lines[1]
                text_lines = lines[2:]
            else:
                time_line = lines[0]
                text_lines = lines[1:]

            # 解析时间轴: 00:00:01,000 --> 00:00:04,000
            match = re.match(
                r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})",
                time_line.strip(),
            )
            if not match:
                continue

            start = BaseSubtitleNode._parse_timestamp(match.group(1))
            end = BaseSubtitleNode._parse_timestamp(match.group(2))
            text = "\n".join(text_lines).strip()

            segments.append({
                "index": idx,
                "start": start,
                "end": end,
                "text": text,
            })

        return segments

    @staticmethod
    def _parse_vtt(content: str) -> list[dict[str, Any]]:
        """解析 WebVTT 格式字幕内容。

        Parameters
        ----------
        content:
            WebVTT 格式字符串。

        Returns
        -------
        list[dict[str, Any]]
            字幕片段列表。
        """
        segments: list[dict[str, Any]] = []
        lines = content.split("\n")

        # 跳过 WEBVTT 头部
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("WEBVTT"):
                start_idx = i + 1
                break

        # 按空行分割块
        remaining = "\n".join(lines[start_idx:])
        blocks = re.split(r"\n\s*\n", remaining.strip())

        idx = 0
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            block_lines = block.split("\n")
            if len(block_lines) < 2:
                continue

            # 第一行可能是序号（VTT 可选）
            time_line_idx = 0
            if not "-->" in block_lines[0]:
                time_line_idx = 1

            if time_line_idx >= len(block_lines):
                continue

            time_line = block_lines[time_line_idx]
            text_lines = block_lines[time_line_idx + 1:]

            match = re.match(
                r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
                time_line.strip(),
            )
            if not match:
                continue

            idx += 1
            start = BaseSubtitleNode._parse_timestamp(match.group(1))
            end = BaseSubtitleNode._parse_timestamp(match.group(2))
            text = "\n".join(text_lines).strip()

            segments.append({
                "index": idx,
                "start": start,
                "end": end,
                "text": text,
            })

        return segments

    @staticmethod
    def _to_srt(segments: list[dict[str, Any]]) -> str:
        """将字幕片段列表转为 SRT 格式字符串。

        Parameters
        ----------
        segments:
            字幕片段列表。

        Returns
        -------
        str
            SRT 格式字符串。
        """
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            start = BaseSubtitleNode._format_timestamp(seg["start"], "srt")
            end = BaseSubtitleNode._format_timestamp(seg["end"], "srt")
            text = seg.get("text", "").strip()
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")  # 空行分隔
        return "\n".join(lines)

    @staticmethod
    def _to_vtt(segments: list[dict[str, Any]]) -> str:
        """将字幕片段列表转为 WebVTT 格式字符串。

        Parameters
        ----------
        segments:
            字幕片段列表。

        Returns
        -------
        str
            WebVTT 格式字符串。
        """
        lines: list[str] = ["WEBVTT", ""]
        for seg in segments:
            start = BaseSubtitleNode._format_timestamp(seg["start"], "vtt")
            end = BaseSubtitleNode._format_timestamp(seg["end"], "vtt")
            text = seg.get("text", "").strip()
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")  # 空行分隔
        return "\n".join(lines)

    @staticmethod
    def _merge_short_segments(
        segments: list[dict[str, Any]],
        min_duration: float = 0.5,
    ) -> list[dict[str, Any]]:
        """合并过短的字幕片段。

        Parameters
        ----------
        segments:
            原始字幕片段列表。
        min_duration:
            最小片段时长（秒），短于此值的片段与下一个合并。

        Returns
        -------
        list[dict[str, Any]]
            合并后的片段列表。
        """
        if not segments:
            return []

        result: list[dict[str, Any]] = []
        current = dict(segments[0])

        for seg in segments[1:]:
            current_duration = current["end"] - current["start"]
            if current_duration < min_duration:
                # 合并到当前
                current["end"] = seg["end"]
                current["text"] = (current.get("text", "") + " " + seg.get("text", "")).strip()
                if "speaker" in seg and "speaker" not in current:
                    current["speaker"] = seg["speaker"]
            else:
                result.append(current)
                current = dict(seg)

        result.append(current)
        return result

    @staticmethod
    def _split_long_segments(
        segments: list[dict[str, Any]],
        max_duration: float = 10.0,
        max_chars: int = 42,
    ) -> list[dict[str, Any]]:
        """拆分过长的字幕片段。

        Parameters
        ----------
        segments:
            原始字幕片段列表。
        max_duration:
            最大片段时长（秒），超过此值的片段按文本拆分。
        max_chars:
            每行最大字符数，用于拆分长文本。

        Returns
        -------
        list[dict[str, Any]]
            拆分后的片段列表。
        """
        result: list[dict[str, Any]] = []
        for seg in segments:
            duration = seg["end"] - seg["start"]
            text = seg.get("text", "").strip()

            if duration <= max_duration and len(text) <= max_chars:
                result.append(seg)
                continue

            # 按标点拆分文本
            sentences = re.split(r"([。！？.!?；;，,\n]+)", text)
            # 重组句子（保留标点）
            parts: list[str] = []
            current_part = ""
            for s in sentences:
                current_part += s
                if any(c in s for c in "。！？.!?；;\n"):
                    if current_part.strip():
                        parts.append(current_part.strip())
                    current_part = ""
            if current_part.strip():
                parts.append(current_part.strip())

            # 如果拆分后只有一段，按字数硬切
            if len(parts) <= 1:
                parts = []
                for i in range(0, len(text), max_chars):
                    parts.append(text[i:i + max_chars])

            # 均分时间轴
            num_parts = len(parts)
            if num_parts == 0:
                result.append(seg)
                continue

            seg_duration = duration / num_parts
            for i, part_text in enumerate(parts):
                new_seg = {
                    "start": seg["start"] + i * seg_duration,
                    "end": seg["start"] + (i + 1) * seg_duration,
                    "text": part_text,
                }
                if "speaker" in seg:
                    new_seg["speaker"] = seg["speaker"]
                result.append(new_seg)

        return result

    # ------------------------------------------------------------------
    # 辅助：构造 SubtitleData
    # ------------------------------------------------------------------
    def _make_subtitle_data(
        self,
        segments: list[dict[str, Any]],
        fmt: str = "srt",
        **metadata: Any,
    ) -> SubtitleData:
        """构造 SubtitleData 实例。

        Parameters
        ----------
        segments:
            字幕片段列表。
        fmt:
            字幕格式。
        **metadata:
            额外元数据。

        Returns
        -------
        SubtitleData
        """
        meta: dict[str, Any] = {"segment_count": len(segments)}
        meta.update(metadata)
        return SubtitleData(
            segments=segments,
            format=fmt,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Node 抽象方法
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def run(self, input_data: MosaicData) -> MosaicData:
        """执行节点逻辑（子类实现）。"""

    def describe(self) -> NodeSpec:
        """返回节点规格说明。"""
        return NodeSpec(
            name=self.name,
            domain=self.domain,
            description=self.description,
            version=self.version,
            input_types=list(self.input_types),
            output_types=list(self.output_types),
            model_info={"type": "subtitle"},
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"<{self.__class__.__name__} name={self.name!r} state={status}>"
        )
