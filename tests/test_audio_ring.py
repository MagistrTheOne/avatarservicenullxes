from __future__ import annotations

import asyncio

import numpy as np
import pytest

from avatar_service.media.audio_ring import AudioRing


@pytest.mark.asyncio
async def test_ring_write_then_read_exact() -> None:
    ring = AudioRing(sample_rate=16_000, capacity_seconds=0.5)
    reader = ring.new_reader(start_at_latest=False)

    pcm = np.arange(-400, 400, dtype=np.int16)
    await ring.write(pcm)
    got = await reader.read_exactly(pcm.size, timeout=0.1)
    assert got is not None
    np.testing.assert_array_equal(got, pcm)


@pytest.mark.asyncio
async def test_multi_reader_independent() -> None:
    ring = AudioRing(sample_rate=16_000, capacity_seconds=0.5)
    r1 = ring.new_reader(start_at_latest=False)
    r2 = ring.new_reader(start_at_latest=False)

    a = np.arange(0, 100, dtype=np.int16)
    b = np.arange(100, 250, dtype=np.int16)
    await ring.write(a)
    await ring.write(b)

    got1 = await r1.read_exactly(a.size + b.size, timeout=0.1)
    got2 = await r2.read_exactly(a.size + b.size, timeout=0.1)
    assert got1 is not None and got2 is not None
    np.testing.assert_array_equal(got1, np.concatenate([a, b]))
    np.testing.assert_array_equal(got2, np.concatenate([a, b]))


@pytest.mark.asyncio
async def test_read_times_out_when_no_data() -> None:
    ring = AudioRing(sample_rate=16_000, capacity_seconds=0.5)
    reader = ring.new_reader(start_at_latest=True)
    got = await reader.read_exactly(100, timeout=0.05)
    assert got is None


@pytest.mark.asyncio
async def test_overflow_drops_oldest() -> None:
    ring = AudioRing(sample_rate=10, capacity_seconds=1.0)  # capacity = 10 samples
    reader = ring.new_reader(start_at_latest=False)
    huge = np.arange(1000, dtype=np.int16)
    await ring.write(huge)
    got = await reader.read_exactly(10, timeout=0.1)
    assert got is not None
    np.testing.assert_array_equal(got, huge[-10:])


@pytest.mark.asyncio
async def test_close_wakes_readers() -> None:
    ring = AudioRing(sample_rate=16_000)
    reader = ring.new_reader(start_at_latest=True)

    async def closer() -> None:
        await asyncio.sleep(0.02)
        await ring.close()

    await asyncio.gather(
        closer(),
        _expect_none(reader),
    )


async def _expect_none(reader: AudioRing.Cursor) -> None:
    got = await reader.read_exactly(10, timeout=1.0)
    assert got is None
