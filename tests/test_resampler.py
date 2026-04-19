from __future__ import annotations

import numpy as np

from avatar_service.media.resampler import PcmResampler


def test_identity_resample_returns_copy() -> None:
    r = PcmResampler(16_000, 16_000)
    pcm = np.arange(-100, 100, dtype=np.int16)
    out = r.process(pcm)
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, pcm)


def test_downsample_48k_to_16k_approx_third_length() -> None:
    r = PcmResampler(48_000, 16_000)
    pcm = np.zeros(48_000, dtype=np.int16)
    out = r.process(pcm)
    # soxr streaming may delay a tail; chunk output should be close to 1/3.
    assert 15_000 <= out.size <= 17_000


def test_upsample_24k_to_48k_doubles_length() -> None:
    r = PcmResampler(24_000, 48_000)
    pcm = np.zeros(2_400, dtype=np.int16)
    out = r.process(pcm)
    assert 4_500 <= out.size <= 5_100
