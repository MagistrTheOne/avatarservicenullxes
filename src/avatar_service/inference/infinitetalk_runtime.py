"""InfiniteTalk runtime adapter.

This backend integrates MeiGen-AI/InfiniteTalk by invoking its official
`generate_infinitetalk.py` CLI for each block and decoding the produced MP4
back into RGB frames for the existing FramePipeline.

Design notes:
- Keep the same `AvatarRuntime` contract used by ArachneRuntime.
- Lazy/external dependency: InfiniteTalk code is expected in `repo_dir`;
  the avatar service process itself does not import wan/torch internals.
- One worker thread per process, mirroring the single-GPU/single-session model.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import av
import numpy as np

from ..logging import get_logger
from .identity_bank import IdentityTokens
from .runtime_base import AvatarRuntime, BlockRequest, GeneratedFrame

_log = get_logger(__name__)

INFINITETALK_AUDIO_SR = 16_000
INFINITETALK_OUTPUT_FPS = 16


class InfiniteTalkRuntime(AvatarRuntime):
    """Runtime that shells out to InfiniteTalk for each block."""

    mode = "infinitetalk"

    def __init__(
        self,
        *,
        repo_dir: str,
        python_bin: str,
        ckpt_dir: str,
        wav2vec_dir: str,
        infinitetalk_model_dir: str,
        quant_dir: str | None,
        size: str,
        sample_steps: int,
        frame_num: int,
        motion_frame: int,
        mode: str,
        text_guidance_scale: float,
        audio_guidance_scale: float,
        temp_dir: str,
    ) -> None:
        self._repo_dir = Path(repo_dir)
        self._python_bin = python_bin
        self._ckpt_dir = ckpt_dir
        self._wav2vec_dir = wav2vec_dir
        self._infinitetalk_model_dir = infinitetalk_model_dir
        self._quant_dir = quant_dir
        self._size = size
        self._sample_steps = int(sample_steps)
        self._frame_num = int(frame_num)
        self._motion_frame = int(motion_frame)
        self._mode = mode
        self._text_guidance_scale = float(text_guidance_scale)
        self._audio_guidance_scale = float(audio_guidance_scale)
        self._temp_dir = Path(temp_dir)

        self._loaded = False
        self._executor: ThreadPoolExecutor | None = None

    async def load(self) -> None:
        if self._loaded:
            return
        script = self._repo_dir / "generate_infinitetalk.py"
        if not self._repo_dir.exists():
            raise FileNotFoundError(f"InfiniteTalk repo dir not found: {self._repo_dir}")
        if not script.exists():
            raise FileNotFoundError(f"InfiniteTalk script missing: {script}")
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infinitetalk-gpu")
        self._loaded = True
        _log.info(
            "infinitetalk.ready",
            repo=str(self._repo_dir),
            size=self._size,
            sample_steps=self._sample_steps,
            frame_num=self._frame_num,
            mode=self._mode,
        )

    async def aclose(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def audio_sample_rate(self) -> int:
        return INFINITETALK_AUDIO_SR

    @property
    def output_fps(self) -> int:
        return INFINITETALK_OUTPUT_FPS

    async def prepare_identity(
        self,
        avatar_key: str,
        reference_image_rgb: np.ndarray,
    ) -> IdentityTokens:
        from PIL import Image

        if reference_image_rgb.dtype != np.uint8:
            reference_image_rgb = reference_image_rgb.astype(np.uint8)
        if reference_image_rgb.ndim != 3 or reference_image_rgb.shape[2] != 3:
            raise ValueError("reference_image_rgb must be (H, W, 3) uint8")
        portrait = Image.fromarray(reference_image_rgb)
        return IdentityTokens(avatar_key=avatar_key, payload=portrait)

    def infer_block(self, request: BlockRequest) -> AsyncIterator[GeneratedFrame]:
        return _InfiniteTalkBlockIterator(self, request)


class _InfiniteTalkBlockIterator:
    _SENTINEL_OK = object()
    _SENTINEL_ERR = object()

    def __init__(self, runtime: InfiniteTalkRuntime, request: BlockRequest) -> None:
        if runtime._executor is None:
            raise RuntimeError("InfiniteTalk runtime is not loaded")
        self._runtime = runtime
        self._request = request
        self._loop = asyncio.get_event_loop()
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=16)
        self._started = False
        self._error: BaseException | None = None
        self._block_index = 0

    def __aiter__(self) -> "_InfiniteTalkBlockIterator":
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

    def _drive_sync(self) -> None:
        runtime = self._runtime
        request = self._request
        run_dir = Path(
            tempfile.mkdtemp(prefix="infinitetalk-block-", dir=str(runtime._temp_dir))
        )
        try:
            image_path = run_dir / "ref.png"
            wav_path = run_dir / "audio.wav"
            input_json_path = run_dir / "input.json"
            output_base = run_dir / "out"
            output_mp4 = Path(f"{output_base}.mp4")

            request.identity_tokens.payload.save(image_path)
            _write_pcm16_wav(wav_path, request.audio_pcm16_16k, sample_rate=INFINITETALK_AUDIO_SR)
            input_json_path.write_text(
                json.dumps(
                    {
                        "prompt": request.prompt,
                        "cond_video": str(image_path),
                        "cond_audio": {"person1": str(wav_path)},
                    }
                ),
                encoding="utf-8",
            )

            cmd = [
                runtime._python_bin,
                "generate_infinitetalk.py",
                "--ckpt_dir",
                runtime._ckpt_dir,
                "--wav2vec_dir",
                runtime._wav2vec_dir,
                "--infinitetalk_dir",
                runtime._infinitetalk_model_dir,
                "--input_json",
                str(input_json_path),
                "--size",
                runtime._size,
                "--sample_steps",
                str(max(2, request.num_inference_steps or runtime._sample_steps)),
                "--mode",
                runtime._mode,
                "--motion_frame",
                str(runtime._motion_frame),
                "--frame_num",
                str(max(13, request.num_frames)),
                "--sample_text_guide_scale",
                str(request.text_guidance_scale or runtime._text_guidance_scale),
                "--sample_audio_guide_scale",
                str(request.audio_guidance_scale or runtime._audio_guidance_scale),
                "--save_file",
                str(output_base),
            ]
            if runtime._quant_dir:
                cmd.extend(["--quant", "fp8", "--quant_dir", runtime._quant_dir])

            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                cwd=str(runtime._repo_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            gen_ms = (time.perf_counter() - t0) * 1000.0
            if proc.returncode != 0:
                raise RuntimeError(
                    "InfiniteTalk command failed: "
                    f"exit={proc.returncode}, stderr={proc.stderr[-2000:]}"
                )
            if not output_mp4.exists():
                raise FileNotFoundError(f"InfiniteTalk output MP4 not found: {output_mp4}")

            frames = _decode_video_rgb_frames(output_mp4)
            if not frames:
                raise RuntimeError("InfiniteTalk produced empty video")

            is_idle = bool(np.max(np.abs(request.audio_pcm16_16k.astype(np.int32))) < 200)
            per_frame_ms = (1000.0 / INFINITETALK_OUTPUT_FPS)
            for idx, frame in enumerate(frames):
                out = GeneratedFrame(
                    image_rgb=frame,
                    frame_index_in_block=idx,
                    block_index=self._block_index,
                    inference_ms=gen_ms if idx == 0 else per_frame_ms,
                    is_idle=is_idle,
                )
                asyncio.run_coroutine_threadsafe(self._queue.put(out), self._loop).result()

            _log.info(
                "infinitetalk.block_done",
                frames=len(frames),
                generation_ms=round(gen_ms, 1),
                output=str(output_mp4),
            )
            asyncio.run_coroutine_threadsafe(self._queue.put(self._SENTINEL_OK), self._loop).result()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            _log.exception("infinitetalk.infer_block.failed", error=str(exc))
            try:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(self._SENTINEL_ERR), self._loop
                ).result()
            except Exception:
                pass
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


def _write_pcm16_wav(path: Path, pcm16: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(pcm16, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def _decode_video_rgb_frames(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is None:
            return frames
        for frame in container.decode(video_stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames

