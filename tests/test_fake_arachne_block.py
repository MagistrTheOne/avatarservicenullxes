from __future__ import annotations

import numpy as np
import pytest

from avatar_service.inference.runtime_base import BlockRequest
from avatar_service.stub.fake_arachne import FakeArachneRuntime


@pytest.mark.asyncio
async def test_fake_runtime_block_yields_expected_count() -> None:
    rt = FakeArachneRuntime(resolution="480p")
    await rt.load()

    identity = await rt.prepare_identity(
        "test_avatar",
        reference_image_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
    )
    samples = int(rt.audio_sample_rate * (25 / rt.output_fps))  # 25 frames worth of audio
    # Inject a sinusoid so the waveform drawing actually has something to show.
    t = np.arange(samples, dtype=np.float32) / rt.audio_sample_rate
    audio = (0.6 * np.sin(2 * np.pi * 220 * t) * 32000).astype(np.int16)

    req = BlockRequest(
        audio_pcm16_16k=audio,
        identity_tokens=identity,
        prompt="A person speaking naturally.",
        num_frames=25,
        num_inference_steps=1,
        resolution="480p",
    )

    frames = []
    async for gf in rt.infer_block(req):
        frames.append(gf)
    assert len(frames) == 25
    assert frames[0].image_rgb.shape == (480, 832, 3)
    # At least one frame should NOT be idle since we fed in a real sine wave.
    assert any(not gf.is_idle for gf in frames)
