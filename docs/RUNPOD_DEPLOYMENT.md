# RunPod Deployment

The avatar service is designed to run as a **persistent** RunPod pod on
H200 hardware. Cold-start of the 13.6B DiT weights is ~30-60 seconds;
serverless is the wrong fit.

## 1. Pod template

| Setting              | Value                                 |
|----------------------|---------------------------------------|
| GPU                  | NVIDIA H200 80/141 GB HBM3e           |
| Container image      | `<your-registry>/avatar-service:latest` (built from `Dockerfile`) |
| Container disk       | 30 GB                                 |
| Volume mount         | `/models` → Network Volume (≥ 50 GB)  |
| Expose HTTP ports    | 8080 (control plane)                  |
| Expose UDP ports     | dynamic (aiortc / WebRTC) — RunPod assigns automatically |
| Env vars             | see `.env.example`                    |

## 2. One-time per Network Volume: pull weights

```bash
# inside the pod terminal
WEIGHTS_DIR=/models ./scripts/download_weights.sh
```

This downloads `MagistrTheOne/ARACHNE-X-ULTRA-AVATAR` into
`/models/ARACHNE-X-ULTRA-AVATAR/`. Subsequent pod starts on the same
volume reuse the cached weights.

## 3. One-time per pod template: install upstream `arachne_x` package

```bash
./scripts/install_arachne_on_pod.sh
```

Clones https://github.com/MagistrTheOne/ARACHNE-X-NULLXES- into
`/opt/arachne-x` and registers it on `sys.path` via a `.pth` file so our
`avatar_service.inference.arachne_runtime` can import it.

## 4. Boot

The Dockerfile's `CMD` is `avatar-service serve`, which starts uvicorn
on `${HTTP_HOST}:${HTTP_PORT}` (defaults `0.0.0.0:8080`). The boot
sequence:

1. Logging is configured (`structlog` JSON in production).
2. The `ArachneRuntime` is instantiated and **scheduled to load**
   asynchronously — `/health` is live immediately, `/health/ready`
   reports `model_loaded: false` until the load + warm-up completes.
3. `GatewayClient` opens an outbound HTTP/2 client to `${GATEWAY_BASE_URL}`
   for `POST /avatar/events`.
4. The session manager is empty until the gateway calls `POST /sessions`.

## 5. Tuning knobs (env vars)

| Variable                        | Default     | Effect |
|---------------------------------|-------------|--------|
| `ARACHNE_RESOLUTION`            | `480p`      | `480p` (832×480) or `720p` (1280×720). 720p ~doubles per-frame compute. |
| `ARACHNE_BLOCK_NUM_FRAMES`      | `93`        | Frames per inference block. Smaller = lower first-frame latency, more frequent denoising overhead. |
| `ARACHNE_NUM_INFERENCE_STEPS`   | `8`         | Distilled fast mode = 4-8; full quality = 16+. Each extra step adds ~30-60 ms per block. |
| `ARACHNE_TEXT_GUIDANCE_SCALE`   | `4.0`       | Higher → tighter prompt adherence. |
| `ARACHNE_AUDIO_GUIDANCE_SCALE`  | `4.0`       | Higher → tighter lip-sync at the cost of motion realism. |
| `ARACHNE_WARMUP_BLOCKS`         | `1`         | Warm-up cycles run on a silent block at boot to amortise CUDA autotune. |
| `ARACHNE_WARMUP_FRAMES`         | `25`        | Frames per warm-up block. Keep small to bound boot time. |
| `OPENAI_REALTIME_MODEL`         | `gpt-realtime` | Override if you want `gpt-realtime-2025-...` etc. |

## 6. Verifying the pod is healthy

```bash
# from anywhere with HTTPS reach to the pod's public URL
curl https://<pod>.runpod.net/health
curl https://<pod>.runpod.net/health/ready
curl https://<pod>.runpod.net/metrics
```

Wait for `health/ready` → `{"ready": true, "model_loaded": true, ...}`
before letting the gateway route a session here.

## 7. Pre-warmed pool (production)

For real interview scheduling you want N≥2 warm pods at all times. The
gateway-side `avatarPoolManager` (separate work item in the gateway repo)
maintains the pool and does sticky session pinning. This avatar service
itself only ever owns one session at a time.
