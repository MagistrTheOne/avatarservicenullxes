<!-- BEGIN:nullxes-avatar-agent-notes -->
# NULLXES / HR AI — Avatar Service agent notes

**Язык:** Python 3.11+, aiortc, httpx, numpy, torch.
**Статус:** отдельный репозиторий (имеет свой `.git`), не subtree в monorepo.
**Целевая платформа:** один H200 GPU = одна pod-сессия. 13.6B DiT, ~120 GB VRAM.

---

## Что это

Единственный процесс, который:

1. Принимает `createSession` от `backend/realtime-gateway/services/avatarClient.ts`.
2. Открывает **два** WebRTC peer'а:
   - `openai_peer` — к OpenAI Realtime API (получает TTS audio + data channel events).
   - `sfu_peer` — к Stream SFU (subscribe mic кандидата, publish `agent_<sessionId>` video + audio).
3. Пересылает аудио OpenAI → video frames → Stream SFU в одном процессе (shared memory, без IPC).
4. Шлёт callback'и в gateway: `POST /avatar/events` (first_frame, session_ready, error, closed).

Детальная архитектура — `docs/ARCHITECTURE.md`.

---

## Карта модулей

### API / запуск
| Файл | За что отвечает |
|---|---|
| `src/avatar_service/api/schemas.py` | Pydantic модели запросов/ответов. |
| `src/avatar_service/api/routes.py` | FastAPI routes (create_session / events / health). |
| `src/avatar_service/main.py` | uvicorn entry. |

### Сессия
| Файл | За что отвечает |
|---|---|
| `src/avatar_service/sessions/avatar_session.py` | `AvatarSession` — держит оба peer'а, `_wait_first_frame(timeout=360.0)`, lifecycle. |
| `src/avatar_service/sessions/registry.py` | sessionId → AvatarSession. |

### Media
| Файл | За что отвечает |
|---|---|
| `src/avatar_service/media/openai_peer.py` | aiortc к OpenAI Realtime. `wait_until_open(timeout=15.0)`. httpx POST multipart (`timeout=30.0`). |
| `src/avatar_service/media/sfu_peer.py` | aiortc к Stream SFU. Stream auth via JWT. httpx POST `timeout=30.0`. |
| `src/avatar_service/media/vision_agents_peer.py` | Альтернативный peer через vision-agents протокол. |
| `src/avatar_service/media/audio_ring.py` | `AudioRing` — 16kHz mono PCM16 shared ring. `read_exactly(count, timeout=...)`. |

### Inference
| Файл | За что отвечает |
|---|---|
| `src/avatar_service/inference/runtime_base.py` | Абстракция over DiT/ARACHNE-X бэкенда. |
| `src/avatar_service/inference/frame_pipeline.py` | Audio chunks → video frames. Timeout = `3 * num_frames / fps` (мин 3с). |

### Encode
| Файл | За что отвечает |
|---|---|
| `src/avatar_service/encode/video_track.py` | aiortc VideoStreamTrack, async queue, `timeout=frame_period * 2`. |

### Bus
| Файл | За что отвечает |
|---|---|
| `src/avatar_service/bus/gateway_client.py` | httpx client к gateway. **`timeout_seconds: float = 5.0` — дефолт.** Callbacks `avatar_event`, `heartbeat`. |

### Docs
- `docs/ARCHITECTURE.md` — схема, audio rings, почему один процесс.
- `docs/API_CONTRACT.md` — JSON-схемы запросов/ответов.

---

## Последнее состояние

- `d2ce566` — `fix(session): bump first-frame timeout to 360s`. До этого 60с не хватало на прогрев DiT.
- `6ad925f` — базовая интеграция ARACHNE-X runtime.

Все `httpx.AsyncClient` в медиа-коде сидят на `timeout=30.0`. Только `bus/gateway_client.py` имеет 5.0 — это для быстрых callback'ов, но **если gateway тормозит — callback теряется**. См. pending.

---

## Gotchas

1. **Один процесс = один H200 GPU = одна сессия.** Не пытайся сделать multi-session в одном процессе — DiT занимает всю VRAM. Горизонтальное масштабирование = несколько pod-реплик.

2. **`_wait_first_frame(timeout=360.0)`** — прогрев модели занимает до 6 минут на холодном старте. Не понижай, не читая логов первого boot.

3. **`openai_peer.wait_until_open(timeout=15.0)`** — если OpenAI data channel не открылся за 15с, считаем fail. Повышать опасно — gateway ждёт `session_ready` callback и валит сессию по своему таймауту раньше.

4. **Shared audio ring** — `AudioRing` не thread-safe для multi-writer. Один писатель (OpenAI peer), много читателей (inference + sfu_peer). НЕ добавляй второй writer.

5. **aiortc + Stream SFU** — Stream ожидает SDP в определённом формате. Наш `sfu_peer` патчит SDP перед отправкой (см. код). Любой апгрейд aiortc тестировать на SDP-совместимость.

6. **`agent_<sessionId>` userId** — обязательный формат. `avatar-stream-card.tsx` во frontend'е whitelist'ит **только** `agent_*` и `agent-<meetingId>`. Если изменишь префикс — HR не увидит аватар.

7. **RunPod warmup** — на cold start весь pod поднимается 2-3 мин. Gateway должен иметь возможность ретраить `createSession` (pending, см. ниже).

---

## Pending

- **`gateway_client.py` timeout 5.0s → 30.0s** — при лагах gateway callback'и теряются. На фронт не влияет (frontend уже не смотрит на `timeout of 5000ms` visually), но heartbeat / first_frame событие могут пропасть и gateway решит что pod мёртв.
- **Graceful shutdown** — сейчас Ctrl+C обрывает активные сессии без callback `closed`. Нужен SIGTERM handler который шлёт события и закрывает peer'ы.
- **Healthcheck** — добавить GPU-memory метрику и latency frame-pipeline в `/health`. Сейчас только liveness.
- **Retry createSession от gateway** — если pod в cold-start, первый POST зафейлится. Пусть gateway ретраит 3× с back-off.

---

## Команды

```bash
cd avatarservicenullxes

# dev
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1 # Windows PowerShell
pip install -r requirements.txt          # CPU
pip install -r requirements-gpu.txt      # + torch CUDA для H200

# run (нужен .env — см. .env.example)
python -m avatar_service.main

# tests
pytest

# docker
docker build -t nullxes/avatar-service .
docker run --gpus all -p 8080:8080 --env-file .env nullxes/avatar-service
```

---

## Контракт с gateway

**Создание сессии** (gateway → pod):
```
POST /sessions
{
  "session_id": "sess_...",
  "meeting_id": "m_...",
  "openai": { "call_id": "...", "client_secret": "...", "peer_sdp_offer": "..." },
  "stream": { "api_key": "...", "call_type": "default", "call_id": "...",
              "agent_token": "<jwt for agent_<sessionId>>" }
}
→ 202 Accepted + session_id
```

**Callback'и** (pod → gateway):
```
POST /avatar/events
{
  "session_id": "...",
  "event": "session_ready" | "first_frame" | "error" | "closed",
  "ts": <unix_ts>,
  "error": { "code": "...", "message": "..." }   // только для error
}
```

Любой mismatch → ломает flow. Следи за `docs/API_CONTRACT.md`.
<!-- END:nullxes-avatar-agent-notes -->
