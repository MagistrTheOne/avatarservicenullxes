# avatarservicenullxes

Real-time AI avatar service for the NULLXES HR AI interview platform.

One Python process on a single NVIDIA **H200** GPU:

- holds a WebRTC peer to **OpenAI Realtime** (TTS audio + DataChannel events);
- runs **ARACHNE-X-ULTRA-AVATAR** (13.6B DiT, `audio + image -> video`) **in-process** — no gRPC, no Triton, no extra hop;
- joins the **Stream SFU** call as a server-side participant `agent_<sessionId>` and publishes both the TTS audio track and the generated avatar video track.

The browser of the candidate talks **only** to the Stream SFU. It never opens a
WebRTC peer to OpenAI, it never relays audio through the gateway. The gateway is
used only for control-plane (session lifecycle, token issuing, SSE for captions
and `avatar_ready` signaling).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design doc.

## Why one process

- Zero network hop between the OpenAI TTS chunk, the ARACHNE inference, and the
  NVENC encoder. Every hop would eat into the sub-300ms mouth-to-display budget.
- Shared monotonic clock for audio publication and video frame PTS — no AV drift.
- One GPU = one session. The DiT model already occupies 110-120 GB of VRAM
  (see the [model card](https://huggingface.co/MagistrTheOne/ARACHNE-X-ULTRA-AVATAR)),
  so multi-tenant sharing is deferred to a later quantization phase.

## High-level flow

```
candidate mic (Stream SFU)
      |
      v
Avatar Service  ---->  OpenAI Realtime  ---->  Avatar Service
                           (TTS audio + oai-events DataChannel)
                                                  |
                                  +---------------+---------------+
                                  |                               |
                                  v                               v
                          ARACHNE-X (in-process)         Delayed audio publish
                               (30 FPS, 33ms/frame)      (for AV sync)
                                  |                               |
                                  v                               v
                          NVENC H.264 encoder                     |
                                  |                               |
                                  +-------->  Stream SFU  <-------+
                                                   |
                                                   v
                                     candidate browser + HR observers
```

## Runtimes

- `ARACHNE_MODE=real` — loads ARACHNE-X-ULTRA-AVATAR weights and runs on a real H200
  (or any CUDA device with enough VRAM and Hopper/Ada support). This is the
  production mode, meant to be deployed on a RunPod persistent pod.
- `ARACHNE_MODE=stub` — loads the `FakeArachneRuntime` which returns a
  reference image overlayed with a live audio waveform. Used for local
  development against the rest of the stack (OpenAI peer, SFU peer, audio
  bridge) without requiring a GPU. The rest of the pipeline is real.

## Quick start (local dev)

```bash
# Python 3.10+
python -m venv .venv
.venv\Scripts\activate         # or source .venv/bin/activate on linux
pip install -e ".[dev]"
cp .env.example .env           # fill in OPENAI_API_KEY, STREAM_*, GATEWAY_*

# local dev — stub mode, no GPU, no model weights
set ARACHNE_MODE=stub
avatar-service serve

# integration smoke tests
python scripts/smoke_openai_peer.py
python scripts/smoke_sfu_peer.py
```

## RunPod deployment

See [docs/RUNPOD_DEPLOYMENT.md](docs/RUNPOD_DEPLOYMENT.md).

Short version:

1. `scripts/download_weights.sh` — pulls ARACHNE-X-ULTRA-AVATAR from Hugging Face into the pod's network volume.
2. Build the Docker image: `docker build -t nullxes/avatar-service:latest .`
3. Push to your RunPod registry, create a persistent pod template with the image,
   a Network Volume mounted at `/models`, and the env vars from `.env.example`.
4. Gateway calls `POST /sessions` on the pod to start an avatar session.

## API

See [docs/API_CONTRACT.md](docs/API_CONTRACT.md) for the full HTTP contract.

Summary:

| Method | Path | Purpose |
|---|---|---|
| `POST`   | `/sessions`           | Start an avatar session (called by the gateway). |
| `DELETE` | `/sessions/{sid}`     | Graceful stop. Frees the pod back to the warm pool. |
| `GET`    | `/sessions/{sid}`     | Inspect current session state. |
| `GET`    | `/health`             | Liveness. |
| `GET`    | `/health/ready`       | Readiness: GPU visible, model loaded, NVENC available. |
| `GET`    | `/metrics`            | Prometheus metrics (Avatar-specific + default `prometheus_client`). |

## License

MIT. See [LICENSE](LICENSE).
