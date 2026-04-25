"""Runtime configuration loaded from environment variables.

Uses pydantic-settings so we get validation, typing, and `.env` autoload for
free. The model is frozen: any `settings` instance returned by `get_settings()`
must be treated as read-only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ArachneMode = Literal["real", "stub", "infinitetalk"]
LogFormat = Literal["pretty", "json"]
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


class Settings(BaseSettings):
    """Strongly-typed settings object. Instantiate via `get_settings()`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Runtime mode ---------------------------------------------------------
    arachne_mode: ArachneMode = "stub"
    arachne_weights_dir: str = "./models/ARACHNE-X-ULTRA-AVATAR"
    arachne_cuda_device: int = 0
    # ARACHNE-X output is 16 FPS (see Demo/run_demo_streaming_realtime.py)
    # and uses "480p" (832x480) or "720p" (1280x720) resolution buckets.
    arachne_resolution: str = "480p"
    # Production block size MUST match `arachne_warmup_frames` so the first
    # live block reuses the CUDA graphs / dynamo specialisations created
    # during warmup. Mismatched shapes trigger a long recompile cascade
    # (`[41/N] q0 is not in var_ranges`) that can take 5+ minutes on the
    # first block.  At 16 FPS, 25 frames = 1.56 s of audio per block,
    # rendered in ~700 ms by H200 — comfortably real-time.
    arachne_block_num_frames: int = 25
    arachne_num_inference_steps: int = 8
    arachne_text_guidance_scale: float = 4.0
    arachne_audio_guidance_scale: float = 4.0
    arachne_warmup_blocks: int = 1
    arachne_warmup_frames: int = 25

    # --- InfiniteTalk runtime --------------------------------------------------
    # Used when ARACHNE_MODE=infinitetalk. This runs a block inference by
    # shelling out to InfiniteTalk's generate_infinitetalk.py script.
    infinitetalk_repo_dir: str = "./third_party/InfiniteTalk"
    infinitetalk_python_bin: str = "python"
    infinitetalk_ckpt_dir: str = "./weights/Wan2.1-I2V-14B-480P"
    infinitetalk_wav2vec_dir: str = "./weights/chinese-wav2vec2-base"
    infinitetalk_model_dir: str = "./weights/InfiniteTalk/single/infinitetalk.safetensors"
    infinitetalk_quant_dir: str = ""
    infinitetalk_size: str = "infinitetalk-480"
    infinitetalk_sample_steps: int = 8
    infinitetalk_frame_num: int = 25
    infinitetalk_motion_frame: int = 9
    infinitetalk_audio_guidance_scale: float = 4.0
    infinitetalk_text_guidance_scale: float = 4.0
    infinitetalk_mode: Literal["clip", "streaming"] = "clip"
    infinitetalk_temp_dir: str = "./tmp/infinitetalk"

    # --- OpenAI Realtime ------------------------------------------------------
    openai_api_key: str = ""
    openai_realtime_model: str = "gpt-realtime"
    openai_realtime_base_url: str = "https://api.openai.com/v1"

    # --- Stream SFU -----------------------------------------------------------
    stream_api_key: str = ""
    stream_api_secret: str = ""
    stream_base_url: str = "https://video.stream-io-api.com"
    # Stream coordinator requires a `location` hint on JoinCall to pick the
    # nearest SFU edge. Frontend SDK derives it from geolocation; the pod is
    # server-side so we use a fixed default. Override per pod region.
    # Common values: amsterdam, frankfurt, london, dublin, oregon, virginia,
    # chicago, mumbai, singapore, tokyo, sydney.
    stream_default_location: str = "amsterdam"
    # Which SFU backend to use:
    # - "vision_agents" (default): official getstream Python SDK with full
    #   coordinator + WS+protobuf SFU JoinFlow + Twirp SetPublisher.
    # - "legacy": hand-rolled REST/Twirp client (kept for emergency fallback;
    #   does NOT work against current Stream prod because they require WS join
    #   before any Twirp request).
    sfu_backend: Literal["vision_agents", "legacy"] = "vision_agents"

    # --- Gateway callback -----------------------------------------------------
    gateway_base_url: str = ""
    gateway_shared_token: str = ""

    # --- HTTP control plane ---------------------------------------------------
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    cors_allowed_origins: str = ""

    # --- Logging --------------------------------------------------------------
    log_level: LogLevel = "info"
    log_format: LogFormat = "pretty"

    # --- Derived --------------------------------------------------------------
    @property
    def cors_allowed_origins_list(self) -> list[str]:
        if not self.cors_allowed_origins:
            return []
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @field_validator("arachne_resolution")
    @classmethod
    def _res_sanity(cls, v: str) -> str:
        if v not in {"480p", "720p"}:
            raise ValueError("arachne_resolution must be '480p' or '720p'")
        return v

    @field_validator("http_port")
    @classmethod
    def _port_sanity(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("http_port must be in [1, 65535]")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    Call this from anywhere; the cache avoids re-parsing `.env` on every call.
    Tests that need to override settings should clear the cache with
    `get_settings.cache_clear()` after mutating the environment.
    """

    return Settings()
