from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor

from .modules.autoencoder_kl_wan import AutoencoderKLWan
from .modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .modules.longcat_video_dit import LongCatVideoTransformer3DModel
from .modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from .audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from .pipeline_arachne_x_video import ArachneXVideoPipeline
from .pipeline_arachne_x_video_avatar import ArachneXVideoAvatarPipeline


@dataclass(frozen=True)
class WeightsLayout:
    tokenizer: str = "tokenizer"
    text_encoder: str = "text_encoder"
    vae: str = "vae"
    scheduler: str = "scheduler"
    dit: str = "dit"
    avatar_single: str = "avatar_single"
    avatar_multi: str = "avatar_multi"
    audio_dir: str = "audio"
    wav2vec2: str = "audio/wav2vec2"
    vocal_separator: str = "audio/vocal_separator/Kim_Vocal_2.onnx"


def _p(root: str, subpath: str) -> str:
    return str(Path(root) / subpath)


def load_base_pipeline(
    checkpoint_dir: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[Tuple[int, int]] = None,
    layout: WeightsLayout = WeightsLayout(),
) -> ArachneXVideoPipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        _p(checkpoint_dir, layout.tokenizer),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        _p(checkpoint_dir, layout.text_encoder),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        _p(checkpoint_dir, layout.vae),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        _p(checkpoint_dir, layout.scheduler),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        _p(checkpoint_dir, layout.dit),
        cp_split_hw=cp_split_hw,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    pipe = ArachneXVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(device)
    return pipe


def load_avatar_pipeline(
    checkpoint_dir: str,
    variant: str = "single",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[Tuple[int, int]] = None,
    layout: WeightsLayout = WeightsLayout(),
) -> ArachneXVideoAvatarPipeline:
    tokenizer = AutoTokenizer.from_pretrained(
        _p(checkpoint_dir, layout.tokenizer),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        _p(checkpoint_dir, layout.text_encoder),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        _p(checkpoint_dir, layout.vae),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        _p(checkpoint_dir, layout.scheduler),
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    if variant not in ("single", "multi"):
        raise ValueError(f"Unknown avatar variant: {variant}")
    avatar_subdir = layout.avatar_single if variant == "single" else layout.avatar_multi
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        _p(checkpoint_dir, avatar_subdir),
        cp_split_hw=cp_split_hw,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    wav2vec_path = _p(checkpoint_dir, layout.wav2vec2)
    audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(device)
    audio_encoder.feature_extractor._freeze_parameters()

    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_path,
        local_files_only=True,
    )

    pipe = ArachneXVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=audio_encoder,
        wav2vec_feature_extractor=wav2vec_feature_extractor,
    )
    pipe.to(device)
    return pipe


def get_vocal_separator_path(checkpoint_dir: str, layout: WeightsLayout = WeightsLayout()) -> str:
    return _p(checkpoint_dir, layout.vocal_separator)
