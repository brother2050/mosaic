# tests/final/test_streaming.py
"""Streaming output global tests.

Tests for the streaming subsystem: StreamSession / StreamAdapter buffer management,
chunk push/pop flow, overlap-add crossfade, callbacks, and backend
``synthesize_stream`` interface verification across all four TTS backends.

Design notes
------------
* Backend ``synthesize_stream`` interface tests (T_STREAM_01~T_STREAM_04) are
  lightweight: they only verify the method exists and is callable on the backend
  class.  They do not require actual model weights or GPU.
* StreamSession tests (T_STREAM_05~T_STREAM_10) are self-contained unit tests
  that exercise the buffer logic directly with synthetic numpy arrays.
"""
from __future__ import annotations

import sys
import math
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, "/workspace/mosaic")

from mosaic.nodes.audio.tts_backends.streaming.base import StreamSession, StreamAdapter
from mosaic.core.types import AudioData


# ============================================================================
# T_STREAM_01 ~ T_STREAM_04: Backend ``synthesize_stream`` interface existence
# ============================================================================
class TestBackendStreamingInterface:
    """Verify that each TTS backend class exposes a ``synthesize_stream`` method."""

    # T_STREAM_01
    def test_T_STREAM_01_chattts_synthesize_stream_callable(self):
        """T_STREAM_01: ChatTTS backend has ``synthesize_stream`` and it is callable."""
        from mosaic.nodes.audio.tts_backends.implementations.chattts_backend import (
            ChatTTSBackend,
        )

        assert hasattr(ChatTTSBackend, "synthesize_stream"), (
            "ChatTTSBackend must expose a 'synthesize_stream' attribute"
        )
        assert callable(getattr(ChatTTSBackend, "synthesize_stream")), (
            "ChatTTSBackend.synthesize_stream must be callable"
        )

    # T_STREAM_02
    def test_T_STREAM_02_fish_synthesize_stream_callable(self):
        """T_STREAM_02: Fish backend has ``synthesize_stream`` and it is callable."""
        from mosaic.nodes.audio.tts_backends.implementations.fish_backend import (
            FishSpeechBackend,
        )

        assert hasattr(FishSpeechBackend, "synthesize_stream"), (
            "FishSpeechBackend must expose a 'synthesize_stream' attribute"
        )
        assert callable(getattr(FishSpeechBackend, "synthesize_stream")), (
            "FishSpeechBackend.synthesize_stream must be callable"
        )

    # T_STREAM_03
    def test_T_STREAM_03_sovits_synthesize_stream_callable(self):
        """T_STREAM_03: GPT-SoVITS backend has ``synthesize_stream`` and it is callable."""
        from mosaic.nodes.audio.tts_backends.implementations.sovits_backend import (
            GPTSoVITSBackend,
        )

        assert hasattr(GPTSoVITSBackend, "synthesize_stream"), (
            "GPTSoVITSBackend must expose a 'synthesize_stream' attribute"
        )
        assert callable(getattr(GPTSoVITSBackend, "synthesize_stream")), (
            "GPTSoVITSBackend.synthesize_stream must be callable"
        )

    # T_STREAM_04
    def test_T_STREAM_04_cosyvoice_synthesize_stream_callable(self):
        """T_STREAM_04: CosyVoice backend has ``synthesize_stream`` and it is callable."""
        from mosaic.nodes.audio.tts_backends.implementations.cosyvoice_backend import (
            CosyVoiceBackend,
        )

        assert hasattr(CosyVoiceBackend, "synthesize_stream"), (
            "CosyVoiceBackend must expose a 'synthesize_stream' attribute"
        )
        assert callable(getattr(CosyVoiceBackend, "synthesize_stream")), (
            "CosyVoiceBackend.synthesize_stream must be callable"
        )


# ============================================================================
# T_STREAM_05: Stream total sample count preservation
# ============================================================================
class TestStreamTotalDuration:
    """Verify that streaming output preserves total sample count."""

    def test_T_STREAM_05_stream_total_samples_matches_input(self):
        """T_STREAM_05: Stream total output sample count equals input sample count.

        Push 5000 samples of a sine wave through StreamSession, pop all chunks,
        flush remaining, and assert the total number of output samples equals 5000.
        """
        chunk_size = 512
        sample_rate = 24000
        session = StreamSession(
            chunk_size=chunk_size,
            overlap=0,
            sample_rate=sample_rate,
        )

        # Generate 5000 samples of a 440 Hz sine wave
        total_input = 5000
        samples = [
            math.sin(2.0 * math.pi * 440.0 * i / sample_rate)
            for i in range(total_input)
        ]
        session.push(np.array(samples, dtype=np.float32))

        # Pop all available chunks
        chunks: list[AudioData] = []
        while True:
            chunk = session.pop()
            if chunk is None:
                break
            chunks.append(chunk)

        # Flush any remaining samples
        final = session.flush()
        if final is not None:
            chunks.append(final)

        assert len(chunks) > 0, (
            "Expected at least one output chunk"
        )

        # Sum the waveform lengths from all chunks
        total_output = 0
        for c in chunks:
            wf = c["waveform"]
            if hasattr(wf, "__len__"):
                total_output += len(wf)

        assert total_output == total_input, (
            f"Total output samples ({total_output}) must equal "
            f"total input samples ({total_input})"
        )


# ============================================================================
# T_STREAM_06: StreamSession push/pop basic flow
# ============================================================================
class TestStreamSessionPushPop:
    """Verify StreamSession push/pop basic flow."""

    def test_T_STREAM_06_push_pop_basic_flow(self):
        """T_STREAM_06: push < chunk_size returns None; push enough yields AudioData."""
        session = StreamSession(chunk_size=100, overlap=0, sample_rate=16000)

        # Push 50 samples -- less than chunk_size, pop should return None
        session.push(np.zeros(50, dtype=np.float32))
        assert session.pop() is None, (
            "pop() must return None when buffer has fewer than chunk_size samples"
        )

        # Push 60 more samples -- now 110 total >= 100 chunk_size
        session.push(np.zeros(60, dtype=np.float32))
        chunk = session.pop()
        assert chunk is not None, (
            "pop() must return an AudioData chunk when buffer >= chunk_size"
        )
        assert isinstance(chunk, AudioData), (
            "pop() must return an AudioData instance"
        )

        # The chunk waveform should have exactly chunk_size samples
        assert len(chunk["waveform"]) == 100, (
            f"First chunk waveform length must be 100, got {len(chunk['waveform'])}"
        )


# ============================================================================
# T_STREAM_07: StreamSession overlap-add smoothness
# ============================================================================
class TestStreamSessionOverlap:
    """Verify StreamSession overlap-add crossfade behaviour."""

    def test_T_STREAM_07_overlap_add_smoothness(self):
        """T_STREAM_07: With overlap > 0, successive pops produce AudioData chunks."""
        session = StreamSession(chunk_size=100, overlap=20, sample_rate=16000)

        # Push 100 samples
        session.push(np.ones(100, dtype=np.float32))
        chunk1 = session.pop()
        assert chunk1 is not None, "First pop must return an AudioData chunk"
        assert isinstance(chunk1, AudioData), (
            "First chunk must be an AudioData instance"
        )

        # Push another 100 samples
        session.push(np.ones(100, dtype=np.float32))
        chunk2 = session.pop()
        assert chunk2 is not None, "Second pop must return an AudioData chunk"
        assert isinstance(chunk2, AudioData), (
            "Second chunk must be an AudioData instance"
        )

        # Both chunks should have correct sample_rate in metadata
        assert chunk1["sample_rate"] == 16000, (
            "chunk1 sample_rate must be 16000"
        )
        assert chunk2["sample_rate"] == 16000, (
            "chunk2 sample_rate must be 16000"
        )


# ============================================================================
# T_STREAM_08: StreamSession on_chunk_ready callback
# ============================================================================
class TestStreamSessionCallback:
    """Verify StreamSession on_chunk_ready callback mechanism."""

    def test_T_STREAM_08_on_chunk_ready_callback(self):
        """T_STREAM_08: Registering a callback triggers it when push fills a chunk."""
        session = StreamSession(chunk_size=100, overlap=0, sample_rate=16000)
        callback_chunks: list[AudioData] = []

        def _callback(chunk: AudioData) -> None:
            callback_chunks.append(chunk)

        session.on_chunk_ready(_callback)

        # Push exactly chunk_size samples -- should trigger the callback
        session.push(np.ones(100, dtype=np.float32))

        assert len(callback_chunks) == 1, (
            f"Callback must be called once, got {len(callback_chunks)} calls"
        )
        assert isinstance(callback_chunks[0], AudioData), (
            "Callback argument must be an AudioData instance"
        )


# ============================================================================
# T_STREAM_09: StreamSession flush outputs remaining samples
# ============================================================================
class TestStreamSessionFlush:
    """Verify StreamSession flush behaviour."""

    def test_T_STREAM_09_flush_outputs_remaining(self):
        """T_STREAM_09: flush() returns remaining samples and sets is_complete."""
        session = StreamSession(chunk_size=100, overlap=0, sample_rate=16000)

        # Push fewer than chunk_size
        session.push(np.ones(60, dtype=np.float32))
        assert session.pop() is None, (
            "pop() must return None when buffer < chunk_size"
        )

        # Flush should return the remaining 60 samples
        remaining = session.flush()
        assert remaining is not None, (
            "flush() must return remaining samples when buffer is non-empty"
        )
        assert isinstance(remaining, AudioData), (
            "flush() must return an AudioData instance"
        )
        assert len(remaining["waveform"]) == 60, (
            f"Remaining waveform length must be 60, got {len(remaining['waveform'])}"
        )
        assert session.is_complete, (
            "is_complete must be True after flush()"
        )


# ============================================================================
# T_STREAM_10: chunk_size parameter affects output granularity
# ============================================================================
class TestStreamSessionChunkSize:
    """Verify chunk_size parameter controls output granularity."""

    def test_T_STREAM_10_chunk_size_affects_granularity(self):
        """T_STREAM_10: Smaller chunk_size produces more chunks than larger chunk_size."""
        total_push = 300

        session_small = StreamSession(chunk_size=100, overlap=0, sample_rate=16000)
        session_large = StreamSession(chunk_size=500, overlap=0, sample_rate=16000)

        session_small.push(np.ones(total_push, dtype=np.float32))
        session_large.push(np.ones(total_push, dtype=np.float32))

        # Pop all chunks from small-chunk session
        small_chunks: list[AudioData] = []
        while True:
            c = session_small.pop()
            if c is None:
                break
            small_chunks.append(c)
        final_small = session_small.flush()
        if final_small is not None:
            small_chunks.append(final_small)

        # Pop all chunks from large-chunk session
        large_chunks: list[AudioData] = []
        while True:
            c = session_large.pop()
            if c is None:
                break
            large_chunks.append(c)
        final_large = session_large.flush()
        if final_large is not None:
            large_chunks.append(final_large)

        assert len(small_chunks) >= len(large_chunks), (
            f"Small chunk_size (100) should produce >= chunks than large (500): "
            f"{len(small_chunks)} vs {len(large_chunks)}"
        )