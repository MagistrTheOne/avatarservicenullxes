"""Run one fake ARACHNE block end-to-end: 25 frames, sine-wave audio.

No GPU required. Saves frames to ``scripts/_fake_block_frames/`` as PNGs
for visual inspection.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from PIL import Image

from avatar_service.inference.runtime_base import BlockRequest
from avatar_service.stub.fake_arachne import FakeArachneRuntime


async def amain() -> int:
    rt = FakeArachneRuntime(resolution="480p")
    await rt.load()
    out_dir = Path(__file__).parent / "_fake_block_frames"
    out_dir.mkdir(exist_ok=True)

    identity = await rt.prepare_identity(
        "ksera_digital_twin",
        reference_image_rgb=np.zeros((0, 0, 3), dtype=np.uint8),
    )
    n_frames = 48  # 3 s at 16 FPS
    samples = int(rt.audio_sample_rate * (n_frames / rt.output_fps))
    t = np.arange(samples, dtype=np.float32) / rt.audio_sample_rate
    audio = (0.6 * np.sin(2 * np.pi * 220 * t) * 32000).astype(np.int16)

    req = BlockRequest(
        audio_pcm16_16k=audio,
        identity_tokens=identity,
        prompt="A person speaking naturally.",
        num_frames=n_frames,
        num_inference_steps=1,
        resolution="480p",
    )
    i = 0
    async for gf in rt.infer_block(req):
        Image.fromarray(gf.image_rgb, mode="RGB").save(out_dir / f"frame_{i:04d}.png")
        i += 1
    print(f"wrote {i} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(amain()))
