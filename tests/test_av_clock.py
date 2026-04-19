from __future__ import annotations

import asyncio

import pytest

from avatar_service.sync.av_clock import AVClock, LatencyEWMA


@pytest.mark.asyncio
async def test_sleep_until_is_approximate() -> None:
    clock = AVClock()
    deadline = clock.now() + 0.05  # 50 ms
    await clock.sleep_until(deadline)
    drift = clock.now() - deadline
    # Windows asyncio sleep can drift up to ~30ms; we just need the clock to
    # not return *before* the deadline.
    assert drift >= -0.005, f"woke up too early: drift={drift*1000:.2f}ms"
    assert drift <= 0.05, f"drift too large: drift={drift*1000:.2f}ms"


def test_latency_ewma_converges() -> None:
    ewma = LatencyEWMA(alpha=0.5)
    for _ in range(50):
        ewma.observe(33.0)
    assert abs(ewma.current_ms - 33.0) < 0.01


def test_latency_percentiles() -> None:
    ewma = LatencyEWMA(alpha=0.1)
    for i in range(100):
        ewma.observe(float(i))
    assert ewma.p50() is not None
    assert ewma.p95() is not None
    assert ewma.p50() <= ewma.p95()
