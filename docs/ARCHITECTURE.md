# avatarservicenullxes — Architecture

## Position in the platform

```
+------------------+     +-----------------+     +------------------+
|  Candidate       |     |  HR Spectator   |     |  Other observers |
|  Browser         |     |                 |     |                  |
+--------+---------+     +--------+--------+     +--------+---------+
         | mic + cam              | subscribe-all         | subscribe-all
         | publish                |                       |
         v                        v                       v
+----------------------------------------------------------------+
|                       Stream Video SFU                         |
|       (one call per interview, multiple participants)          |
+--------+----------------------+--------------------------+-----+
         | candidate mic         | agent audio + avatar     |
         | (subscribe)           | video (publish as        |
         v                       | agent_<sessionId>)       |
+--------+--------+              v                          |
|  Avatar pod     +<-----------------------------------------+
|  (one H200,     |     subscribe & publish via aiortc
|   one process)  |
|                 |     +------------------------------------+
|  +-----------+  +<--->|  OpenAI Realtime API               |
|  | OpenAI    |  |     |  (TTS audio + DataChannel events)  |
|  | peer      |  |     +------------------------------------+
|  +-----------+  |
|  +-----------+  |     +------------------------------------+
|  | ARACHNE-X |  +---->|  NULLXES gateway                   |
|  | in-proc   |  |     |  POST /avatar/events,              |
|  +-----------+  |     |  GET /interviews/:id/events (SSE)  |
|  +-----------+  |     +------------------------------------+
|  | aiortc    |  |
|  | SFU peer  |  |
|  +-----------+  |
+-----------------+
```

## Why one process

- Inference and SFU publishing share a monotonic clock — no AV drift.
- Audio rings sit in shared memory; no IPC, no extra hop.
- The 13.6B DiT pegs ~120 GB VRAM on a single H200, which is exactly one
  GPU per session anyway. Splitting into multiple services would buy
  nothing and cost a network hop on every frame.

## Audio rings

Two `AudioRing` instances at **16 kHz mono PCM16**:

| Ring        | Producer                          | Consumers                                          |
|-------------|-----------------------------------|----------------------------------------------------|
| `mic_ring`  | `StreamSfuPeer._drain_candidate_audio` (resampled from 48 kHz) | `OpenAIRealtimePeer._MicForwardTrack` (re-ups to 24 kHz inside the OpenAI handshake) |
| `tts_ring`  | `OpenAIRealtimePeer._drain_remote_audio` (resampled from 24 kHz) | `StreamSfuPeer._AgentAudioTrack` (resamples to 48 kHz on output) and `FramePipeline` (consumes float32 16 kHz directly for ARACHNE) |

16 kHz is the lowest common denominator: it's what Wav2Vec2 expects natively
and the 24↔48 kHz resampling is cheap on either end. Crucially, the same
TTS audio that the SFU publishes to the candidate's ears also drives the
ARACHNE inference — guaranteed lip-sync because there is exactly one
source.

## Frame pipeline

```
TTS chunks (16 kHz)  ->  tts_ring  ->  FramePipeline (asyncio task)
                                          |
                                          v
                                     BlockRequest
                                       (audio block,
                                        identity tokens,
                                        prompt, emotion)
                                          |
                                          v
                              ARACHNE.infer_block(...)
                                  (yields frames)
                                          |
                                          v
                               AvatarVideoTrack.push()
                                          |
                                          v
                                   aiortc -> SFU
```

ARACHNE is **block-streaming**, not per-frame: one call generates
`num_frames` (default 93) frames at 16 FPS. The frame pipeline issues
back-to-back blocks; cross-block continuity comes from the audio
conditioning (the public DiT API isn't stateful between calls).

## Latency

Per ARACHNE-X-NULLXES- README the target on a single H200 is **33 ms per
frame after the first** (streaming VAE decode); the full denoising loop
for one block of 93 frames @ 8 distilled steps takes ~500-900 ms. End-to-
end mouth-to-display latency on a candidate browser then roughly breaks
down as:

| Hop                                          | Typical |
|----------------------------------------------|---------|
| OpenAI TTS chunk arrives at avatar pod       | 25-60 ms (us-east) |
| Decode Opus + write ring                     | 1-2 ms |
| Wait for next block worth of audio (~5.8 s) — overlaps with inference of the previous block | – |
| ARACHNE block first-frame latency            | 500-900 ms |
| ARACHNE per-frame after first                | 30-50 ms |
| NVENC / aiortc encode + packetize            | 5-10 ms |
| SFU forward                                  | 8-25 ms |
| Browser jitter buffer + render               | 50-110 ms |

So **first speech in a new block** lands at the candidate roughly 700-1100
ms after OpenAI starts producing it, and subsequent frames within the
block trail by a steady ~150-250 ms. This is dominated by the diffusion
model's own latency, not by transport. Future work (TensorRT compile,
CUDA graphs, INT8) directly attacks the per-block first-frame number;
see `docs/RUNPOD_DEPLOYMENT.md` for tuning knobs.

## Failure modes

- **Audio underrun**: the frame pipeline waits up to 3× block duration for
  audio. If still empty, it injects a silence block and ARACHNE renders an
  idle face (Motion Decoupling). The session keeps running.
- **OpenAI peer dies**: surfaces as `__transport.closed` event on the
  internal bus, which the session forwards to the gateway as `error`.
  The gateway is responsible for stopping the session and notifying the
  user.
- **SFU peer dies**: same handling — the session shuts down cleanly and
  surfaces an `error` event.
- **Inference failure**: bubbles up through the `_BlockIterator` and
  causes the session to transition to `failed`.

## Rationale for the chosen runtime API

The upstream pipeline at
https://github.com/MagistrTheOne/ARACHNE-X-NULLXES- exposes
`pipe.generate_streaming_ai2v(image, prompt, audio_stream, resolution,
num_frames, num_inference_steps, ...)` as the official Audio + Image →
Video streaming entry point (see `Demo/run_demo_streaming_realtime.py`
in that repo). We bind to it directly inside `inference/arachne_runtime.py`
and translate our `BlockRequest` into its arguments without any
intermediate process, so any optimizations the upstream applies (CUDA
graphs, `torch.compile`, BSA attention, distilled scheduler) apply
unmodified.
