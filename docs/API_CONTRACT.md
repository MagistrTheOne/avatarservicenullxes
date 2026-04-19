# API Contract

The HTTP control plane is the only surface the gateway talks to. Everything
else (OpenAI, Stream SFU) is consumed inside the avatar pod.

## `POST /sessions` — create a session

Called by the gateway when a meeting starts. Returns `202 Accepted` and
the session begins booting in the background. The gateway should poll
`GET /sessions/{sid}` or wait for the `avatar_ready` event delivered to
its own `POST /avatar/events` endpoint.

### Request

```json
{
  "meeting_id": "meeting_24",
  "session_id": "sess_24_2026-04-18T07:42:00Z",
  "avatar_key": "ksera_digital_twin",
  "transport": "webrtc-sfu",
  "openai": {
    "instructions": "You are a senior HR interviewer ...",
    "voice": "alloy",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "input_audio_transcription_model": "gpt-4o-mini-transcribe",
    "temperature": 0.6
  },
  "sfu": {
    "call_type": "default",
    "call_id": "meeting_24",
    "agent_user_id": "agent_sess_24_2026-04-18T07:42:00Z",
    "agent_user_name": "HR ассистент",
    "agent_user_token": "<JWT minted by the gateway>",
    "candidate_user_id": "candidate_24"
  },
  "arachne": {
    "prompt": "A senior HR specialist speaking calmly, neutral office background.",
    "resolution": "480p",
    "num_frames": 93,
    "num_inference_steps": 8,
    "text_guidance_scale": 4.0,
    "audio_guidance_scale": 4.0
  },
  "reference_image": {
    "url": "https://cdn.nullxes.com/avatars/ksera/portrait_512.jpg",
    "sha256": "9f2a...c41e"
  },
  "emotion": "warm"
}
```

#### Field notes

- **`openai.instructions`** is the **LLM system prompt** (long, multi-KB, drives what the agent says).
- **`arachne.prompt`** is a **short text condition for the DiT** (visual style / context). Keep under ~200 chars; the AVATAR model card recommends including a `speaking` / `talking` token.
- **`reference_image`** is the AI2V reference portrait (face crop, ~512 px on the long side, JPEG/PNG, ≤8 MB). Provide either `url` (HTTPS) or `base64` (raw bytes; `data:` URI prefix is auto-stripped); never both. Optional `sha256` is verified after fetch. The pod caches the encoded identity tokens by `avatar_key` in an LRU; subsequent sessions with the same key skip the re-encode.
- **`arachne.resolution`** = `480p` (832×480) or `720p` (1280×720). 480p is recommended for the 16 FPS interview track.

### Response (`202 Accepted`)

```json
{
  "provider": "runpod",
  "session_id": "sess_24_2026-04-18T07:42:00Z",
  "status": "starting",
  "agent_user_id": "agent_sess_24_2026-04-18T07:42:00Z"
}
```

### Errors

- `409 pod_busy` — another session is already active on this pod.
- `503 runtime_not_loaded` — the model is still loading; retry shortly.
- `422 reference_image_invalid` — fetch / decode / SHA-256 mismatch.
- `4xx` — Pydantic validation failure on the request body.

## `GET /sessions/{sid}` — inspect

Returns the full `SessionSnapshot` (see `api/schemas.py`):

```json
{
  "session_id": "sess_24_2026-04-18T07:42:00Z",
  "meeting_id": "meeting_24",
  "phase": "ready",
  "agent_user_id": "agent_sess_24_2026-04-18T07:42:00Z",
  "created_at": 1745000400.123,
  "ready_at": 1745000401.987,
  "stopped_at": null,
  "last_error": null,
  "frames_published": 1248,
  "audio_chunks_published": 0,
  "inference_latency_ms_p50": 33.1,
  "inference_latency_ms_p95": 51.7,
  "audio_underruns": 0
}
```

## `DELETE /sessions/{sid}` — graceful stop

Asks the avatar service to stop the session. Returns the final
`SessionSnapshot` once the pipeline has shut down.

## `GET /health` and `GET /health/ready`

`/health` is always 200 once uvicorn is up. `/health/ready` reflects the
deeper state — model loaded, GPU visible, NVENC available, no active
session in a failure phase. Pod-level health probes should hit
`/health/ready`.

## `GET /metrics`

Prometheus exposition. Per-session counters bubble through:
`avatar_active_sessions`, `avatar_model_loaded`, `avatar_frames_published_total`.

## Outbound: `POST /avatar/events` on the gateway

The avatar pod posts JSON envelopes shaped like:

```json
{
  "type": "avatar_ready",
  "session_id": "sess_24_2026-04-18T07:42:00Z",
  "meeting_id": "meeting_24",
  "ts": 1745000401.987,
  "data": {
    "agent_user_id": "agent_sess_24_2026-04-18T07:42:00Z",
    "resolution": "480p",
    "fps": 16
  }
}
```

`type` is one of:

| Type                    | Meaning |
|-------------------------|---------|
| `avatar_ready`          | First frame published to the SFU. The gateway should mirror this onto its SSE stream so the candidate browser swaps the placeholder for the live participant tile. |
| `transcript_delta`      | Streaming chunk of OpenAI transcript for either `assistant` or `candidate` (see `data.role`). |
| `transcript_completed`  | End of an utterance. |
| `response_done`         | OpenAI finished a `response.create`. |
| `error`                 | Something failed; `data.where` localises it. |
| `stopped`               | Graceful stop completed. |

The `Authorization: Bearer ${GATEWAY_SHARED_TOKEN}` header is set on every
call. The gateway should reject requests without it.
