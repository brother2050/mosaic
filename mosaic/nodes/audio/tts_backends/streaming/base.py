# mosaic/nodes/audio/tts_backends/streaming/base.py
"""TTS 流式适配层 —— Layer 4。

管理音频缓冲区、chunk 切分与实时输出。``StreamSession`` 使用环形缓冲区
（``collections.deque``）累积推理线程推入的波形，并按固定 ``chunk_size``
切分为输出 chunk；``StreamAdapter`` 作为工厂创建和管理流式会话。

设计要点
--------
* 线程安全：推理线程 ``push``、输出线程 ``pop``，通过 ``threading.Lock``
  串行化所有缓冲区访问。
* overlap-add 交叉淡化：每个输出 chunk 的前 ``overlap`` 样本与上一个 chunk
  的末尾 ``overlap`` 样本做线性插值（``fade_out * prev_tail + fade_in * new``），
  平滑 chunk 边界。
* numpy 惰性导入：模块加载不依赖 numpy，仅在数据处理时导入。
* 回调驱动 / 拉取驱动二选一：注册 ``on_chunk_ready`` 后，``push`` 会自动
  输出整 chunk；否则由消费方主动 ``pop``。
"""

from __future__ import annotations

import collections
import threading
from collections.abc import Callable
from typing import Any

from mosaic.core.types import AudioData

__all__ = ["StreamAdapter", "StreamSession"]


class StreamSession:
    """流式会话，管理一个合成任务的音频缓冲区。

    使用环形缓冲区管理音频数据，支持 overlap-add 平滑处理。
    线程安全：支持在推理线程中 ``push``、在输出线程中 ``pop``。

    Parameters
    ----------
    chunk_size : int
        每个输出 chunk 的样本数，默认 4096。
    overlap : int
        chunk 间的重叠样本数（用于交叉淡化），默认 256。
    sample_rate : int
        采样率。
    buffer_dtype : str
        缓冲区数据类型，默认 ``"float32"``。
    """

    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 256,
        sample_rate: int = 24000,
        buffer_dtype: str = "float32",
    ) -> None:
        self._chunk_size = int(chunk_size)
        self._overlap = int(overlap)
        self._sample_rate = int(sample_rate)
        self._buffer_dtype = buffer_dtype
        # 环形缓冲区（存 float 值）
        self._buffer: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()
        self._chunks_emitted = 0
        self._total_samples = 0
        self._is_complete = False
        self._callback: Callable[[AudioData], None] | None = None
        # 上一个 chunk 的末尾 overlap 样本（用于交叉淡化）
        self._prev_tail: Any = None

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def chunks_emitted(self) -> int:
        """已输出的 chunk 数量。"""
        return self._chunks_emitted

    @property
    def total_samples(self) -> int:
        """已缓冲的总样本数。"""
        return self._total_samples

    @property
    def is_complete(self) -> bool:
        """流是否已完成（flush 后）。"""
        return self._is_complete

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def push(self, waveform_chunk: Any) -> None:
        """向缓冲区推入一段波形。

        Parameters
        ----------
        waveform_chunk : numpy.ndarray 或 list
            一段音频波形数据（1D float 数组）。
        """
        import numpy as np  # 惰性导入

        emitted: list[AudioData] = []
        with self._lock:
            # 将数据转换为 numpy array（1D float）
            arr = np.asarray(waveform_chunk, dtype=self._buffer_dtype)
            arr = np.atleast_1d(arr).ravel()
            # 扩展到缓冲区 deque
            self._buffer.extend(arr.tolist())
            self._total_samples += int(arr.shape[0])
            # 如果缓冲区数据 >= chunk_size，自动触发回调（如果有）
            if self._callback is not None:
                while len(self._buffer) >= self._chunk_size:
                    chunk = self._pop_unlocked()
                    if chunk is None:
                        break
                    emitted.append(chunk)
        # 在锁外调用回调，避免回调中再次访问会话造成死锁
        for chunk in emitted:
            self._try_callback(chunk)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def pop(self) -> AudioData | None:
        """从缓冲区取出一个 chunk。

        如果缓冲区数据 >= ``chunk_size``，返回一个 chunk；
        否则返回 ``None``（缓冲区不足）。

        如果有 ``overlap``，新 chunk 的前 ``overlap`` 样本与上一个 chunk
        的末尾 ``overlap`` 样本做线性插值（交叉淡化）。

        Returns
        -------
        AudioData | None
            一个音频 chunk；``None`` 表示缓冲区不足。
        """
        with self._lock:
            return self._pop_unlocked()

    def _pop_unlocked(self) -> AudioData | None:
        """``pop`` 的无锁核心实现，调用者须持有 ``self._lock``。"""
        # 如果缓冲区数据 < chunk_size 且未完成，返回 None
        if not self._is_complete and len(self._buffer) < self._chunk_size:
            return None
        if len(self._buffer) == 0:
            return None
        # 从缓冲区取出 chunk_size 个样本（不足时取全部）
        n = min(self._chunk_size, len(self._buffer))
        samples = [self._buffer.popleft() for _ in range(n)]
        return self._build_chunk(samples)

    def flush(self) -> AudioData | None:
        """强制输出缓冲区中剩余的所有样本。

        在合成完成后调用。输出剩余的不足一个 chunk 的数据。

        Returns
        -------
        AudioData | None
            剩余的音频；``None`` 表示无剩余数据。
        """
        with self._lock:
            # 设置 is_complete = True
            self._is_complete = True
            # 输出缓冲区中所有剩余样本（不足 chunk_size 也要输出）
            if len(self._buffer) == 0:
                return None
            n = len(self._buffer)
            samples = [self._buffer.popleft() for _ in range(n)]
            return self._build_chunk(samples)

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def on_chunk_ready(self, callback: Callable[[AudioData], None]) -> None:
        """注册回调：每当有 chunk 可用时自动调用。

        Parameters
        ----------
        callback : Callable[[AudioData], None]
            回调函数，接收一个 :class:`AudioData` 参数。
        """
        self._callback = callback

    def _try_callback(self, chunk: AudioData) -> None:
        """如果有回调，自动调用。"""
        if self._callback is not None:
            self._callback(chunk)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _build_chunk(self, samples: list[float]) -> AudioData:
        """将原始样本列表封装为输出 chunk。

        包含交叉淡化、保存末尾 overlap 样本、计数更新与 ``AudioData`` 构造。
        调用者须持有 ``self._lock``。
        """
        import numpy as np  # 惰性导入

        chunk = np.asarray(samples, dtype=self._buffer_dtype)
        # overlap-add 交叉淡化
        chunk = self._apply_crossfade(chunk)
        # 保存当前 chunk 的末尾 overlap 样本，供下一次交叉淡化
        if self._overlap > 0 and chunk.shape[0] >= self._overlap:
            self._prev_tail = chunk[-self._overlap:].copy()
        elif self._overlap > 0 and chunk.shape[0] > 0:
            self._prev_tail = chunk.copy()
        # 更新 chunks_emitted
        self._chunks_emitted += 1
        # 构造 AudioData 返回
        return AudioData(
            waveform=chunk,
            sample_rate=self._sample_rate,
            metadata={
                "chunk_index": self._chunks_emitted - 1,
                "chunk_size": int(chunk.shape[0]),
                "sample_rate": self._sample_rate,
                "is_final": self._is_complete and len(self._buffer) == 0,
            },
        )

    def _apply_crossfade(self, new_chunk: Any) -> Any:
        """对新 chunk 的前 ``overlap`` 样本与上一个 chunk 的末尾做交叉淡化。

        使用线性插值：``fade_out * prev_tail + fade_in * new``。
        """
        import numpy as np  # 惰性导入

        # 如果没有 prev_tail 或 overlap == 0，直接返回
        if self._prev_tail is None or self._overlap <= 0:
            return new_chunk
        overlap = min(self._overlap, new_chunk.shape[0], self._prev_tail.shape[0])
        if overlap <= 0:
            return new_chunk
        # 否则做线性交叉淡化
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=new_chunk.dtype)
        fade_out = 1.0 - fade_in
        out = new_chunk.copy()
        out[:overlap] = (
            fade_out * self._prev_tail[:overlap] + fade_in * out[:overlap]
        )
        return out


class StreamAdapter:
    """流式适配器，创建和管理流式会话。

    Parameters
    ----------
    chunk_size : int
        每个输出 chunk 的样本数，默认 4096。
    overlap : int
        chunk 间的重叠样本数，默认 256。
    sample_rate : int
        采样率，默认 24000。
    buffer_dtype : str
        缓冲区数据类型，默认 ``"float32"``。
    """

    def __init__(
        self,
        chunk_size: int = 4096,
        overlap: int = 256,
        sample_rate: int = 24000,
        buffer_dtype: str = "float32",
    ) -> None:
        self._chunk_size = int(chunk_size)
        self._overlap = int(overlap)
        self._sample_rate = int(sample_rate)
        self._buffer_dtype = buffer_dtype

    def create_stream(self, total_samples_hint: int | None = None) -> StreamSession:
        """创建一个新的流式会话。

        Parameters
        ----------
        total_samples_hint : int | None
            预计总样本数（用于预分配缓冲区，可选）。

        Returns
        -------
        StreamSession
            新的流式会话实例。
        """
        # total_samples_hint 当前作为预留参数；deque 无需预分配即可高效增长。
        return StreamSession(
            chunk_size=self._chunk_size,
            overlap=self._overlap,
            sample_rate=self._sample_rate,
            buffer_dtype=self._buffer_dtype,
        )
