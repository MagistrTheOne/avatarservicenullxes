"""Real ARACHNE-X-ULTRA-AVATAR runtime.

Binds this service to the upstream package published at
https://github.com/MagistrTheOne/ARACHNE-X-NULLXES- .

Contract
--------
We use the official AI2V streaming entry point::

    from arachne_x.loader import load_avatar_pipeline
    pipe = load_avatar_pipeline(
        checkpoint_dir="/models/ARACHNE-X-ULTRA-AVATAR",
        variant="single",
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    for frame_np in pipe.generate_streaming_ai2v(
        image=PIL_reference_portrait,
        prompt=text_prompt,
        audio_stream=audio_chunks_16k,   # generator yielding np.float32 chunks
        resolution="480p",               # or "720p"
        num_frames=93,
        num_inference_steps=8,           # distilled fast mode
        text_guidance_scale=4.0,
        audio_guidance_scale=4.0,
    ):
        # frame_np: [H, W, 3] uint8, 16 FPS timeline
        ...

ARACHNE operates in **block streaming mode**: one call generates an entire
block (``num_frames`` frames @ 16 FPS, ~5.8 s for 93 frames) and streams
frames out of the VAE decode as soon as denoising is done. The frame pipeline
on top of this file drives multiple back-to-back blocks; cross-block
continuity comes from the audio conditioning alone (the DiT is not stateful
across calls in the public API).

Audio in/out
------------
ARACHNE's Wav2Vec2 encoder is locked to **16 kHz mono float32**. Our OpenAI
peer delivers 24 kHz PCM16 and the Stream SFU delivers 48 kHz Opus; a
resampler upstream of this runtime (``media/resampler.py``) converts to
16 kHz before handing us the audio.

Import policy
-------------
`arachne_x` is **not** installed in the dev container — it only exists on the
RunPod pod once the upstream repo is checked out and `pip install -e .` has
been run. We import it lazily inside :meth:`load` so the stub runtime stays
importable on developer laptops.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..logging import get_logger
from .identity_bank import IdentityTokens
from .runtime_base import AvatarRuntime, BlockRequest, GeneratedFrame

_log = get_logger(__name__)

ARACHNE_AUDIO_SR = 16_000
ARACHNE_OUTPUT_FPS = 16


class ArachneRuntime(AvatarRuntime):
    """Production backend — loads ARACHNE-X weights on the configured CUDA device."""

    mode = "real"

    def __init__(
        self,
        weights_dir: str,
        cuda_device: int,
        resolution: str = "480p",
        warmup_blocks: int = 1,
        warmup_frames: int = 25,
        default_num_inference_steps: int = 8,
    ) -> None:
        self._weights_dir = Path(weights_dir)
        self._cuda_device = int(cuda_device)
        self._resolution = resolution
        self._warmup_blocks = int(warmup_blocks)
        self._warmup_frames = int(warmup_frames)
        self._default_steps = int(default_num_inference_steps)

        self._loaded = False
        self._executor: ThreadPoolExecutor | None = None
        self._pipe: Any = None
        self._device: Any = None

    # ------------------------------------------------------------- lifecycle
    async def load(self) -> None:
        if self._loaded:
            return
        if not self._weights_dir.exists():
            raise FileNotFoundError(f"ARACHNE weights dir not found: {self._weights_dir}")
        # Single worker: one inference in flight per GPU.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arachne-gpu")

        _log.info("arachne.loading", weights=str(self._weights_dir), device=self._cuda_device)
        t0 = time.perf_counter()
        await asyncio.get_running_loop().run_in_executor(self._executor, self._load_sync)
        _log.info("arachne.loaded", elapsed_sec=round(time.perf_counter() - t0, 2))

        if self._warmup_blocks > 0:
            await self._run_warmup()

        self._loaded = True
        _log.info("arachne.ready")

    def _load_sync(self) -> None:
        """Synchronous model load — runs in the worker thread."""

        import torch  # local import so the stub runtime doesn't require torch

        from arachne_x.loader import load_avatar_pipeline  # type: ignore[import-not-found]
        from arachne_x.streaming_inference import CUDAOptimizer  # type: ignore[import-not-found]

        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            raise RuntimeError("ARACHNE_MODE=real but no CUDA device is visible to torch")
        torch.cuda.set_device(self._cuda_device)
        self._device = torch.device(f"cuda:{self._cuda_device}")

        CUDAOptimizer.enable_flash_attention()

        self._pipe = load_avatar_pipeline(
            checkpoint_dir=str(self._weights_dir),
            variant="single",
            device=f"cuda:{self._cuda_device}",
            torch_dtype=torch.bfloat16,
        )
        # Freeze: we never train at runtime.
        if hasattr(self._pipe, "eval"):
            self._pipe.eval()

    async def _run_warmup(self) -> None:
        """Run a couple of silent-audio blocks so CUDA autotune is done before first real audio."""

        from PIL import Image

        _log.info("arachne.warmup.start", blocks=self._warmup_blocks, frames=self._warmup_frames)
        portrait = Image.new("RGB", (512, 512), color=(128, 128, 128))
        identity = IdentityTokens(avatar_key="__warmup__", payload=portrait)
        silence_samples = int(ARACHNE_AUDIO_SR * (self._warmup_frames / ARACHNE_OUTPUT_FPS))
        silence = np.zeros(silence_samples, dtype=np.int16)
        for i in range(self._warmup_blocks):
            req = BlockRequest(
                audio_pcm16_16k=silence,
                identity_tokens=identity,
                prompt="A person speaking naturally.",
                num_frames=self._warmup_frames,
                num_inference_steps=max(4, self._default_steps // 2),
                resolution=self._resolution,
            )
            frame_count = 0
            async for _ in self.infer_block(req):
                frame_count += 1
            _log.info("arachne.warmup.block", index=i, frames=frame_count)

    async def aclose(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._pipe = None
        self._loaded = False
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------- props
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def audio_sample_rate(self) -> int:
        return ARACHNE_AUDIO_SR

    @property
    def output_fps(self) -> int:
        return ARACHNE_OUTPUT_FPS

    # ------------------------------------------------------------- identity
    async def prepare_identity(
        self,
        avatar_key: str,
        reference_image_rgb: np.ndarray,
    ) -> IdentityTokens:
        """Prepare identity tokens for the given portrait.

        The upstream pipeline accepts `PipelineImageInput` (PIL.Image /
        numpy / torch tensor) directly in ``generate_streaming_ai2v``. To
        keep the bank's payload lightweight (and to avoid pinning a GPU
        tensor into the cache), we persist a PIL.Image here and let the
        pipeline consume it each call. Per-call preprocessing on the H200
        is sub-millisecond.
        """

        from PIL import Image

        if reference_image_rgb.dtype != np.uint8:
            reference_image_rgb = reference_image_rgb.astype(np.uint8)
        if reference_image_rgb.ndim != 3 or reference_image_rgb.shape[2] != 3:
            raise ValueError("reference_image_rgb must be (H, W, 3) uint8")
        portrait = Image.fromarray(reference_image_rgb)
        return IdentityTokens(avatar_key=avatar_key, payload=portrait)

    # ------------------------------------------------------------- inference
    def infer_block(self, request: BlockRequest) -> AsyncIterator[GeneratedFrame]:
        return _BlockIterator(self, request)


class _BlockIterator:
    """Bridges the synchronous ARACHNE generator into an asyncio iterator.

    The ARACHNE pipeline yields frames from a worker thread. We push them
    into an ``asyncio.Queue`` from that thread using the loop's thread-safe
    scheduler; the iterator pulls from the queue on the event loop side.
    """

    _SENTINEL_OK = object()
    _SENTINEL_ERR = object()

    def __init__(self, runtime: ArachneRuntime, request: BlockRequest) -> None:
        if runtime._executor is None or runtime._pipe is None:
            raise RuntimeError("arachne runtime is not loaded")
        self._runtime = runtime
        self._request = request
        self._loop = asyncio.get_event_loop()
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=16)
        self._started = False
        self._block_index = 0  # future work: track across blocks at session level
        self._error: BaseException | None = None

    def __aiter__(self) -> "_BlockIterator":
        return self

    async def __anext__(self) -> GeneratedFrame:
        if not self._started:
            self._started = True
            assert self._runtime._executor is not None
            self._runtime._executor.submit(self._drive_sync)

        item = await self._queue.get()
        if item is self._SENTINEL_OK:
            raise StopAsyncIteration
        if item is self._SENTINEL_ERR:
            assert self._error is not None
            raise self._error
        return cast(GeneratedFrame, item)

    # -------- running inside the GPU worker thread --------

    def _drive_sync(self) -> None:
        pipe = self._runtime._pipe
        request = self._request

        def _audio_stream_generator():
            """Yield one chunk with the full block audio as float32."""
            pcm_f32 = request.audio_pcm16_16k.astype(np.float32) / 32768.0
            yield pcm_f32

        frame_index = 0
        block_index = self._block_index
        try:
            t0 = time.perf_counter()
            gen = pipe.generate_streaming_ai2v(
                image=request.identity_tokens.payload,
                prompt=request.prompt,
                audio_stream=_audio_stream_generator(),
                resolution=request.resolution,
                num_frames=request.num_frames,
                num_inference_steps=request.num_inference_steps,
                text_guidance_scale=request.text_guidance_scale,
                audio_guidance_scale=request.audio_guidance_scale,
                **_extra_kwargs(request),
            )
            last = time.perf_counter()
            for frame_np in gen:
                now = time.perf_counter()
                per_frame_ms = (now - last) * 1000.0
                last = now
                is_idle = bool(np.max(np.abs(request.audio_pcm16_16k.astype(np.int32))) < 200)
                out = GeneratedFrame(
                    image_rgb=np.asarray(frame_np, dtype=np.uint8),
                    frame_index_in_block=frame_index,
                    block_index=block_index,
                    inference_ms=per_frame_ms if frame_index > 0 else (now - t0) * 1000.0,
                    is_idle=is_idle,
                )
                frame_index += 1
                # Push onto the asyncio queue from this worker thread.
                asyncio.run_coroutine_threadsafe(self._queue.put(out), self._loop).result()
            asyncio.run_coroutine_threadsafe(self._queue.put(self._SENTINEL_OK), self._loop).result()
        except BaseException as exc:  # noqa: BLE001 — propagate any GPU failure to the loop
            self._error = exc
            _log.exception("arachne.infer_block.failed", error=str(exc))
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(self._SENTINEL_ERR), self._loop
                ).result()
            except Exception:  # pragma: no cover
                pass


def _extra_kwargs(request: BlockRequest) -> dict[str, Any]:
    """Build the optional extension kwargs (identity/emotion) accepted by generate_streaming_ai2v."""

    kwargs: dict[str, Any] = {}
    if request.emotion:
        kwargs["emotion_id"] = request.emotion
        if request.emotion_intensity:
            kwargs["emotion_intensity"] = float(request.emotion_intensity)
    return kwargs
