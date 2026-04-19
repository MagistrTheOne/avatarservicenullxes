"""
ARACHNE-X Distilled Scheduler Configuration
Optimized for real-time inference (8 steps vs 50)
"""

import warnings

import numpy as np
import torch


class DistilledSchedulerConfig:
    """Configuration for fast 8-step distilled inference."""
    
    # Timestep schedule for 8-step inference
    TIMESTEP_SCHEDULE_8STEP = torch.tensor([
        999, 875, 750, 625, 500, 375, 250, 125
    ], dtype=torch.long)
    
    # Alternative 4-step ultra-fast schedule
    TIMESTEP_SCHEDULE_4STEP = torch.tensor([
        999, 666, 333, 0
    ], dtype=torch.long)
    
    @staticmethod
    def get_sigmas_for_steps(num_steps: int = 8) -> torch.Tensor:
        """
        Compute sigma schedule (noise levels) for N steps.
        Follows linear interpolation from 1.0 -> 0.001.
        """
        sigmas = torch.linspace(1, 0.001, num_steps)
        return sigmas
    
    @staticmethod
    def get_timesteps(num_steps: int = 8) -> torch.Tensor:
        """Get timesteps for distilled inference."""
        if num_steps == 8:
            return DistilledSchedulerConfig.TIMESTEP_SCHEDULE_8STEP
        elif num_steps == 4:
            return DistilledSchedulerConfig.TIMESTEP_SCHEDULE_4STEP
        else:
            # Linear interpolation for arbitrary step counts
            indices = torch.linspace(0, 999, num_steps, dtype=torch.long)
            return indices
    
    @staticmethod
    def scale_latents_for_distill(latents: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Optional: Scale latents for distilled inference.
        Some distilled models benefit from latent scaling.
        """
        return latents * scale


class TorchCompileConfig:
    """Torch.compile optimization settings for H200."""
    
    # Mode options: "default", "reduce-overhead", "max-autotune"
    MODE = "reduce-overhead"  # Best for inference latency
    
    # Full graph compilation
    FULLGRAPH = False
    
    # Backend: "inductor", "aot_ts_nvfuser", "cudagraph"
    BACKEND = "inductor"


class QuantizationConfig:
    """Quantization settings for extreme optimization."""

    # INT8 quantization (dynamic quant in streaming_inference.QuantizationUtils.quantize_to_int8).
    USE_INT8 = False

    # FP8: not implemented for DiT in this repo; quantize_to_fp8() raises NotImplementedError.
    # Setting USE_FP8 True only marks intent for external tooling — it does not enable FP8 inference here.
    USE_FP8 = False

    # KV-cache compression dtype
    KV_CACHE_DTYPE = torch.float16  # float16 by default; torch.float8_e4m3fn etc. are experimental/off-repo

    @staticmethod
    def warn_if_fp8_requested(use_fp8: bool) -> None:
        if use_fp8:
            warnings.warn(
                "QuantizationConfig requested USE_FP8=True, but FP8 DiT quantization is not "
                "implemented in ARACHNE-X (see streaming_inference.QuantizationUtils.quantize_to_fp8). "
                "Use INT8 or full precision.",
                UserWarning,
                stacklevel=2,
            )


class MemoryOptimizationConfig:
    """Memory efficiency settings."""
    
    # Enable gradient checkpointing
    USE_GRADIENT_CHECKPOINTING = False  # Inference only, not needed
    
    # Enable activation checkpointing
    USE_ACTIVATION_CHECKPOINTING = True
    
    # KV-cache offload to CPU
    OFFLOAD_KV_TO_CPU = True
    
    # KV-cache offload to NVMe (requires nvme_path)
    OFFLOAD_KV_TO_NVME = False
    NVME_PATH = "/mnt/nvme/kv_cache"
    
    # Batch size for single frame decode
    VAE_DECODE_BATCH_SIZE = 1


class StreamingConfig:
    """Real-time streaming configuration."""
    
    # Audio chunk duration (seconds)
    AUDIO_CHUNK_DURATION = 0.5
    
    # VAE decode chunk size (frames)
    VAE_CHUNK_SIZE = 1
    
    # KV-cache window size (frames to keep in cache)
    KV_CACHE_WINDOW_SIZE = 13
    
    # Enable async audio prefetch
    ENABLE_AUDIO_PREFETCH = True
    
    # Audio prefetch buffer size
    AUDIO_PREFETCH_BUFFER_SIZE = 5
    
    # Enable streaming output (yield frames as they're ready)
    ENABLE_STREAMING_OUTPUT = True


def get_realtime_config(target_fps: int = 30, hardware: str = "H200") -> dict:
    """
    Get optimized configuration for real-time inference.
    
    Args:
        target_fps: Target frames per second (30 is ideal for H200).
        hardware: "H200", "H100", "A100".
    
    Returns:
        Config dict with all optimizations.
    """
    
    if target_fps >= 30:
        num_inference_steps = 8  # Distilled
        quant = QuantizationConfig()
        quant.USE_FP8 = False
    elif target_fps >= 15:
        num_inference_steps = 12
        quant = QuantizationConfig()
        quant.USE_FP8 = False
    else:
        num_inference_steps = 20
        quant = QuantizationConfig()
        quant.USE_INT8 = True
    
    QuantizationConfig.warn_if_fp8_requested(quant.USE_FP8)

    config = {
        "num_inference_steps": num_inference_steps,
        "torch_compile": {
            "mode": TorchCompileConfig.MODE,
            "fullgraph": TorchCompileConfig.FULLGRAPH,
        },
        "quantization": {
            "use_int8": quant.USE_INT8,
            "use_fp8": quant.USE_FP8,
            "kv_cache_dtype": str(quant.KV_CACHE_DTYPE),
        },
        "memory_optimization": {
            "use_activation_checkpointing": MemoryOptimizationConfig.USE_ACTIVATION_CHECKPOINTING,
            "offload_kv_to_cpu": MemoryOptimizationConfig.OFFLOAD_KV_TO_CPU,
            "vae_decode_batch_size": MemoryOptimizationConfig.VAE_DECODE_BATCH_SIZE,
        },
        "streaming": {
            "audio_chunk_duration": StreamingConfig.AUDIO_CHUNK_DURATION,
            "vae_chunk_size": StreamingConfig.VAE_CHUNK_SIZE,
            "kv_cache_window_size": StreamingConfig.KV_CACHE_WINDOW_SIZE,
            "enable_audio_prefetch": StreamingConfig.ENABLE_AUDIO_PREFETCH,
            "enable_streaming_output": StreamingConfig.ENABLE_STREAMING_OUTPUT,
        },
        "target_fps": target_fps,
        "hardware": hardware,
    }
    
    return config


# Export as ready-to-use config
REALTIME_CONFIG_30FPS = get_realtime_config(target_fps=30, hardware="H200")
REALTIME_CONFIG_15FPS = get_realtime_config(target_fps=15, hardware="H200")
REALTIME_CONFIG_QUALITY = get_realtime_config(target_fps=8, hardware="H200")
