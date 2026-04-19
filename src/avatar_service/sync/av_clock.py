"""Monotonic clock + latency EWMA for AV synchronization.

The inference loop drives both audio publication and video frame PTS from a
single `AVClock` so we never drift. The `LatencyEWMA` tracks recent inference
latency so we can delay audio publication by the right amount — if we push
audio to the SFU as soon as OpenAI hands it to us, the lips will always trail
the voice by ~33 ms.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass


class AVClock:
    """Monotonic clock anchored at the moment of construction.

    `now()` returns seconds since anchor. `sleep_until(deadline)` yields the
    event loop until `now()` >= `deadline`, using `asyncio.sleep` for waits
    longer than 1 ms and a tight busy-wait for sub-millisecond precision.
    """

    def __init__(self) -> None:
        self._anchor = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._anchor

    async def sleep_until(self, deadline: float) -> None:
        remaining = deadline - self.now()
        if remaining <= 0:
            return
        if remaining > 0.001:
            await asyncio.sleep(remaining - 0.0005)
        # tight final spin for the last ~500us
        while self.now() < deadline:
            await asyncio.sleep(0)

    def reset(self) -> None:
        self._anchor = time.monotonic()


@dataclass
class _LatencySample:
    value_ms: float
    at: float


class LatencyEWMA:
    """Exponential-weighted moving average of a latency stream, in milliseconds.

    Also keeps a bounded sample window so we can surface p50/p95 without
    running a full percentile over the whole history.
    """

    def __init__(self, alpha: float = 0.1, window: int = 1024) -> None:
        self.alpha = alpha
        self._value: float | None = None
        self._samples: deque[_LatencySample] = deque(maxlen=window)

    def observe(self, value_ms: float, at: float | None = None) -> None:
        if at is None:
            at = time.monotonic()
        if self._value is None:
            self._value = value_ms
        else:
            self._value = (1 - self.alpha) * self._value + self.alpha * value_ms
        self._samples.append(_LatencySample(value_ms, at))

    @property
    def current_ms(self) -> float:
        return self._value or 0.0

    def percentile_ms(self, pct: float) -> float | None:
        if not self._samples:
            return None
        xs = sorted(s.value_ms for s in self._samples)
        k = max(0, min(len(xs) - 1, int(round(pct / 100.0 * (len(xs) - 1)))))
        return xs[k]

    def p50(self) -> float | None:
        return self.percentile_ms(50)

    def p95(self) -> float | None:
        return self.percentile_ms(95)

    def reset(self) -> None:
        self._value = None
        self._samples.clear()
