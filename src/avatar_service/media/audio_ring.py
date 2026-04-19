"""Async ring buffer for 16-bit mono PCM.

The OpenAI peer writes 20-40 ms TTS chunks as they arrive; the inference loop
reads fixed-size frame-aligned chunks (~33 ms @ 24 kHz => ~800 samples) at
30 FPS; the TTS publisher reads larger chunks (20 ms @ 48 kHz => 960 samples)
for Opus packing.

All readers contend for the same ring — `AudioRing` is safe for one writer and
many readers, each with an independent cursor. Each reader is modelled as a
`Cursor` object that owns its read position.

The ring drops data on overflow (keeps the most recent N samples) rather than
blocking the writer — audio underrun in the reader is recoverable by padding
with silence, whereas stalling the OpenAI peer is not.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np

PCM_DTYPE = np.int16


@dataclass
class _Reader:
    """Independent cursor into the ring."""

    # Global monotonic sample index this reader is at. Equal to `write_index` - lag.
    read_index: int = 0


class AudioRing:
    """Single-writer, multi-reader ring buffer of PCM samples.

    Parameters
    ----------
    sample_rate
        Sample rate of the PCM stored here. All samples written must match.
    capacity_seconds
        Ring size in seconds. 1 second is plenty for a 30 FPS pipeline; older
        data is discarded on overflow.
    """

    def __init__(self, sample_rate: int, capacity_seconds: float = 1.0) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self._capacity = max(int(sample_rate * capacity_seconds), sample_rate // 10)
        self._buffer = np.zeros(self._capacity, dtype=PCM_DTYPE)
        self._write_index = 0  # global monotonic sample counter
        self._readers: list[_Reader] = []
        self._not_empty = asyncio.Condition()
        self._closed = False

    # ------------------------------------------------------------------ reader

    def new_reader(self, start_at_latest: bool = True) -> "AudioRing.Cursor":
        """Create a new independent reader.

        `start_at_latest=True` positions the reader at the current write head;
        `False` positions it at whatever history is still in the ring.
        """

        reader = _Reader(read_index=self._write_index if start_at_latest else max(0, self._write_index - self._capacity))
        self._readers.append(reader)
        return AudioRing.Cursor(self, reader)

    # ------------------------------------------------------------------ writer

    async def write(self, samples: np.ndarray) -> None:
        """Append samples to the ring. Never blocks; overwrites the oldest data."""

        if self._closed:
            return
        if samples.dtype != PCM_DTYPE:
            samples = samples.astype(PCM_DTYPE)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        n = samples.size
        if n == 0:
            return

        cap = self._capacity
        if n >= cap:
            # Only the last `cap` samples survive.
            samples = samples[-cap:]
            n = cap
            self._buffer[:] = samples
            self._write_index += cap
        else:
            start = self._write_index % cap
            end = start + n
            if end <= cap:
                self._buffer[start:end] = samples
            else:
                first = cap - start
                self._buffer[start:] = samples[:first]
                self._buffer[: n - first] = samples[first:]
            self._write_index += n

        async with self._not_empty:
            self._not_empty.notify_all()

    # -------------------------------------------------------- internal helpers

    def _read_copy(self, reader: _Reader, count: int) -> np.ndarray | None:
        """Return up to `count` samples starting at reader's cursor. Moves the cursor."""

        available = self._write_index - reader.read_index
        if available <= 0:
            return None
        # If the reader has fallen behind more than `capacity`, fast-forward it
        # to the oldest valid sample to avoid reading garbage.
        if available > self._capacity:
            reader.read_index = self._write_index - self._capacity
            available = self._capacity

        take = min(available, count)
        cap = self._capacity
        start = reader.read_index % cap
        end = start + take
        if end <= cap:
            out = self._buffer[start:end].copy()
        else:
            first = cap - start
            out = np.empty(take, dtype=PCM_DTYPE)
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[: take - first]
        reader.read_index += take
        return out

    # ------------------------------------------------------------------ close

    async def close(self) -> None:
        self._closed = True
        async with self._not_empty:
            self._not_empty.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def write_index(self) -> int:
        return self._write_index

    # ------------------------------------------------------------------ cursor

    class Cursor:
        """Independent reader cursor on an `AudioRing`."""

        def __init__(self, ring: "AudioRing", reader: _Reader) -> None:
            self._ring = ring
            self._reader = reader

        async def read_exactly(self, count: int, timeout: float | None = None) -> np.ndarray | None:
            """Block until exactly `count` samples are available; return them.

            Returns None if the ring closes or the timeout elapses with fewer
            samples than requested.
            """

            deadline = asyncio.get_running_loop().time() + timeout if timeout is not None else None
            collected: list[np.ndarray] = []
            remaining = count
            while remaining > 0:
                chunk = self._ring._read_copy(self._reader, remaining)
                if chunk is not None and chunk.size > 0:
                    collected.append(chunk)
                    remaining -= chunk.size
                    continue
                if self._ring.closed:
                    return None
                async with self._ring._not_empty:
                    wait_for: float | None
                    if deadline is not None:
                        wait_for = max(0.0, deadline - asyncio.get_running_loop().time())
                        if wait_for == 0.0:
                            return None
                    else:
                        wait_for = None
                    try:
                        if wait_for is None:
                            await self._ring._not_empty.wait()
                        else:
                            await asyncio.wait_for(self._ring._not_empty.wait(), timeout=wait_for)
                    except asyncio.TimeoutError:
                        return None
            return np.concatenate(collected) if collected else np.zeros(0, dtype=PCM_DTYPE)

        def try_read(self, count: int) -> np.ndarray | None:
            """Non-blocking read of up to `count` samples. Returns None if none are available."""

            return self._ring._read_copy(self._reader, count)

        @property
        def lag_samples(self) -> int:
            return self._ring._write_index - self._reader.read_index
