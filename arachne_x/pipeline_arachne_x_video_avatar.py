import os
import types
from typing import Any, Dict, List, Optional, Union, Literal, Tuple

import gc
import time
import math
import torch
import torch.nn as nn
import loguru
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from tqdm import tqdm 
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import PipelineImageInput
from transformers import AutoTokenizer, UMT5EncoderModel

from arachne_x.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from arachne_x.modules.autoencoder_kl_wan import AutoencoderKLWan
from arachne_x.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from arachne_x.context_parallel import context_parallel_util
from arachne_x.utils.bukcet_config import get_bucket_config

import ftfy
import regex as re
import html

# -------- avatar related --------
import scipy.signal as ss
import pyloudnorm as pyln
from arachne_x.audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from arachne_x.audio_process.multi_stream_processor import MultiStreamAudioProcessor
from arachne_x.audio_process.phoneme_aligner import PhonemeTemporalAligner
from arachne_x.utils.monitoring import MetricsLogger, sha256_of_audio_array
from arachne_x.streaming_inference import StreamingVAEDecoder, CUDAOptimizer
from transformers import Wav2Vec2FeatureExtractor
from diffusers.image_processor import is_valid_image, is_valid_image_imagelist
import warnings


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class GenerationInterrupted(Exception):
    """
    Raised to abort diffusion denoising loops during realtime interruptions.

    This avoids silent partial updates and reduces wasted compute.
    """


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

def preprocess_video(self, video, height: Optional[int] = None, width: Optional[int] = None, resize_mode: Optional[str] = 'crop') -> torch.Tensor:
    r"""
    hack diffusers.video_processor.VideoProcessor to support the parameter of resize_mode 
    """
    if isinstance(video, list) and isinstance(video[0], np.ndarray) and video[0].ndim == 5:
        warnings.warn(
            "Passing `video` as a list of 5d np.ndarray is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 5d np.ndarray",
            FutureWarning,
        )
        video = np.concatenate(video, axis=0)
    if isinstance(video, list) and isinstance(video[0], torch.Tensor) and video[0].ndim == 5:
        warnings.warn(
            "Passing `video` as a list of 5d torch.Tensor is deprecated."
            "Please concatenate the list along the batch dimension and pass it as a single 5d torch.Tensor",
            FutureWarning,
        )
        video = torch.cat(video, axis=0)

    # ensure the input is a list of videos:
    # - if it is a batch of videos (5d torch.Tensor or np.ndarray), it is converted to a list of videos (a list of 4d torch.Tensor or np.ndarray)
    # - if it is a single video, it is converted to a list of one video.
    if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim == 5:
        video = list(video)
    elif isinstance(video, list) and is_valid_image(video[0]) or is_valid_image_imagelist(video):
        video = [video]
    elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
        video = video
    else:
        raise ValueError(
            "Input is in incorrect format. Currently, we only support numpy.ndarray, torch.Tensor, PIL.Image.Image"
        )

    video = torch.stack([self.preprocess(img, height=height, width=width, resize_mode=resize_mode) for img in video], dim=0)
    video = video.permute(0, 2, 1, 3, 4)

    return video

class LongCatVideoAvatarPipeline:
    r"""
    Pipeline for text-to-video generation using LongCatVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        dit: LongCatVideoAvatarTransformer3DModel,
        audio_encoder: Wav2Vec2ModelWrapper,
        wav2vec_feature_extractor: Wav2Vec2FeatureExtractor
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.dit = dit 
        self.device = "cuda"

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8 
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.video_processor.preprocess_video = types.MethodType(preprocess_video, self.video_processor)

        self._num_timesteps = 1000
        self._num_distill_sample_steps = 50

        self.audio_encoder=audio_encoder
        self.wav2vec_feature_extractor = wav2vec_feature_extractor
        # audio processing and monitoring
        self.audio_processor = MultiStreamAudioProcessor()
        self.multi_stream_fusion_proj = nn.Linear(1024, 768)
        self.multi_stream_fusion_scale = 0.2
        # Step 3: phoneme-conditioned stream with robust wav2vec fallback.
        self.phoneme_enabled = True
        self.phoneme_num_classes = 10
        self.phoneme_stream_scale = 0.20
        self.phoneme_confidence_floor = 0.10
        self.phoneme_fallback_to_wav2vec = False
        self.phoneme_aligner = PhonemeTemporalAligner(num_phonemes=self.phoneme_num_classes)
        audio_embed_dim = 768
        self.phoneme_proj = nn.Sequential(
            nn.Linear(self.phoneme_num_classes, audio_embed_dim),
            nn.SiLU(),
            nn.Linear(audio_embed_dim, audio_embed_dim),
        )
        self.phoneme_alignment_head = nn.Linear(audio_embed_dim, self.phoneme_num_classes)
        # Step 4: explicit emotion control channel with lip-sync safety guard.
        self.emotion_enabled = True
        self.emotion_num_classes = 8
        self.emotion_default_id = 0
        self.emotion_default_intensity = 0.0
        self.emotion_lipsync_guard_ratio = 0.35
        self.emotion_label_to_id = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "surprised": 4,
            "fearful": 5,
            "disgusted": 6,
            "calm": 7,
        }
        self.emotion_embedding = nn.Embedding(self.emotion_num_classes, audio_embed_dim)
        nn.init.normal_(self.emotion_embedding.weight, mean=0.0, std=0.02)
        self.emotion_proj = nn.Sequential(
            nn.Linear(audio_embed_dim, audio_embed_dim),
            nn.SiLU(),
            nn.Linear(audio_embed_dim, audio_embed_dim),
        )
        # Step 5: hybrid renderer for controlled mouth zone.
        self.hybrid_renderer_enabled = True
        self.hybrid_renderer_mouth_strength = 0.35
        self.hybrid_renderer_blur_passes = 2
        self.hybrid_renderer_temporal_alpha = 0.70
        self.hybrid_renderer_flicker_budget = 1.40
        self.hybrid_renderer_artifact_budget = 0.08
        self.metrics = MetricsLogger()

        # Identity token bank (Step 2): learnable per-identity vectors injected
        # into text conditioning as extra tokens.
        self.identity_bank_enabled = True
        self.identity_bank_size = 1024
        self.identity_tokens_per_id = 4
        self.identity_token_dim = int(self.text_encoder.config.d_model)
        self.identity_embedding = nn.Embedding(
            self.identity_bank_size,
            self.identity_tokens_per_id * self.identity_token_dim,
        )
        nn.init.normal_(self.identity_embedding.weight, mean=0.0, std=0.02)
        latent_dim = int(getattr(self.vae.config, "z_dim", 16))
        self.identity_latent_projector = nn.Sequential(
            nn.Linear(latent_dim, self.identity_token_dim),
            nn.SiLU(),
            nn.Linear(self.identity_token_dim, self.identity_tokens_per_id * self.identity_token_dim),
        )
        self.identity_default_strength = 1.0
        self.identity_default_negative_strength = 0.0
        
        self.streaming_enabled = True
        # Temporal compression memory (Step 1): keep a recent sliding window and
        # summarize older conditioning frames inside KV-cache for long-context AVC.
        self.temporal_memory_enabled = True
        self.temporal_memory_window_frames = 8
        self.temporal_memory_summary_frames = 2
        self._emotion_guidance_scale = 0.0
        
        # CUDA optimizations for H200
        CUDAOptimizer.enable_flash_attention()
        if hasattr(torch, 'compile'):
            self.dit = CUDAOptimizer.compile_model(self.dit, mode='reduce-overhead')
            self.vae = CUDAOptimizer.compile_model(self.vae, mode='reduce-overhead')

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        mask = mask.to(device=device)
        if num_videos_per_prompt > 1:
            mask = mask.repeat_interleave(num_videos_per_prompt, dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)

        return prompt_embeds, mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
            
        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        scale_factor_spatial
    ):
        # Check height and width divisibility
        if height % scale_factor_spatial != 0 or width % scale_factor_spatial != 0:
            raise ValueError(f"`height and width` have to be divisible by {scale_factor_spatial} but are {height} and {width}.")

        # Check prompt validity
        if prompt is None:
            raise ValueError("Cannot leave `prompt` undefined.")
        
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt has to be of type str or list` but is {type(prompt)}")
        
        # Check negative prompt validity
        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt has to be of type str or list` but is {type(negative_prompt)}")
        
    def prepare_latents(
        self,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        num_cond_frames_added: int = 0,
        need_encode: bool = True
    ) -> torch.Tensor:
        if (image is not None) and (video is not None):
            raise ValueError("Cannot provide both `image and video` at the same time. Please provide only one.")
        if latents is not None:
            latents = latents.to(device=device, dtype=dtype)
        else:
            num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # Generate random noise with shape latent_shape
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        if image is not None or video is not None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )
            condition_data = image if image is not None else video
            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal

            if need_encode:
                is_image = image is not None
                cond_latents = []
                for i in range(batch_size):
                    gen = generator[i] if isinstance(generator, list) else generator
                    if is_image:
                        encoded_input = condition_data[i].unsqueeze(0).unsqueeze(2)
                    else:
                        encoded_input = condition_data[i][:, -(num_cond_frames-num_cond_frames_added):].unsqueeze(0)
                    if num_cond_frames_added > 0:
                        pad_front = encoded_input[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
                        encoded_input = torch.cat([pad_front, encoded_input], dim=2)
                    assert encoded_input.shape[2] == num_cond_frames
                    latent = retrieve_latents(self.vae.encode(encoded_input), gen, sample_mode="argmax")
                    cond_latents.append(latent)

                cond_latents = torch.cat(cond_latents, dim=0).to(dtype)
                cond_latents = self.normalize_latents(cond_latents)
            else:
                cond_latents = condition_data[:, :, -num_cond_latents:]
            
            latents[:, :, :num_cond_latents] = cond_latents

        return latents

    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale
    
    @property
    def audio_guidance_scale(self):
        return self._audio_guidance_scale

    @property
    def emotion_guidance_scale(self):
        return self._emotion_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return (
            self._text_guidance_scale > 1.0
            or self._audio_guidance_scale > 1.0
            or self._emotion_guidance_scale > 0.0
        )

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def num_distill_sample_steps(self):
        return self._num_distill_sample_steps
    
    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs
    
    def get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool=False):
        if use_distill:
            distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32)
            distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()
            
            inference_indices = np.linspace(0, self.num_distill_sample_steps, num=sampling_steps, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)
            
            sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / self.num_timesteps
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps)
        sigmas = sigmas.to(torch.float32)
        return sigmas

    def _update_kv_cache_dict(self, kv_cache_dict):
        self.kv_cache_dict = kv_cache_dict

    def _refresh_identity_tokens(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        identity_id: Optional[Union[int, List[int], torch.Tensor]],
        identity_strength: float,
        identity_negative_strength: float,
        batch_size: int,
        num_videos_per_prompt: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.identity_bank_enabled or identity_id is None:
            return prompt_embeds, negative_prompt_embeds
        if prompt_embeds is None or prompt_embeds.shape[2] < self.identity_tokens_per_id:
            return prompt_embeds, negative_prompt_embeds

        base_ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)
        expanded_ids: List[int] = []
        for idx in base_ids:
            expanded_ids.extend([idx] * num_videos_per_prompt)

        id_index = torch.tensor(
            expanded_ids,
            dtype=torch.long,
            device=self.identity_embedding.weight.device,
        )
        refreshed_tokens = self.identity_embedding(id_index).view(
            len(expanded_ids),
            self.identity_tokens_per_id,
            self.identity_token_dim,
        ).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype).unsqueeze(1)
        prompt_embeds[:, :, -self.identity_tokens_per_id :, :] = refreshed_tokens * float(identity_strength)

        if (
            negative_prompt_embeds is not None
            and negative_prompt_embeds.shape[2] >= self.identity_tokens_per_id
        ):
            negative_prompt_embeds[:, :, -self.identity_tokens_per_id :, :] = refreshed_tokens.to(
                device=negative_prompt_embeds.device,
                dtype=negative_prompt_embeds.dtype,
            ) * float(identity_negative_strength)

        return prompt_embeds, negative_prompt_embeds

    def _predict_avatar_noise(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        audio_embs: torch.Tensor,
        num_cond_latents: Optional[int] = None,
        kv_cache_dict: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        num_ref_latents: int = 0,
        ref_img_index: Optional[int] = None,
        mask_frame_range: Optional[int] = None,
        ref_target_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "audio_embs": audio_embs,
        }
        if num_cond_latents is not None:
            kwargs["num_cond_latents"] = num_cond_latents
        if kv_cache_dict is not None:
            kwargs["kv_cache_dict"] = kv_cache_dict
        if num_ref_latents:
            kwargs["num_ref_latents"] = num_ref_latents
        if ref_img_index is not None:
            kwargs["ref_img_index"] = ref_img_index
        if mask_frame_range is not None:
            kwargs["mask_frame_range"] = mask_frame_range
        if ref_target_masks is not None:
            kwargs["ref_target_masks"] = ref_target_masks
        # torch.compile + CUDAGraphs can reuse internal output buffers across
        # sequential CFG passes (uncond/text/audio), so we mark a fresh step and
        # detach the returned tensor from any reusable graph-managed storage.
        compiler_ns = getattr(torch, "compiler", None)
        if compiler_ns is not None and hasattr(compiler_ns, "cudagraph_mark_step_begin"):
            compiler_ns.cudagraph_mark_step_begin()

        noise_pred = self.dit(**kwargs)
        if isinstance(noise_pred, torch.Tensor):
            return noise_pred.clone()
        return noise_pred

    def _normalize_identity_ids(
        self,
        identity_id: Optional[Union[int, List[int], torch.Tensor]],
        batch_size: int,
    ) -> Optional[List[int]]:
        if identity_id is None:
            return None

        if isinstance(identity_id, int):
            ids = [identity_id] * batch_size
        elif isinstance(identity_id, torch.Tensor):
            ids = [int(x) for x in identity_id.detach().cpu().view(-1).tolist()]
        elif isinstance(identity_id, (list, tuple)):
            ids = [int(x) for x in identity_id]
        else:
            raise TypeError(
                f"`identity_id` must be int, list[int], torch.Tensor, or None. Got {type(identity_id)}."
            )

        if len(ids) == 1 and batch_size > 1:
            ids = ids * batch_size
        if len(ids) != batch_size:
            raise ValueError(
                f"`identity_id` length must be 1 or equal to batch size ({batch_size}), got {len(ids)}."
            )

        for idx in ids:
            if idx < 0 or idx >= self.identity_bank_size:
                raise ValueError(
                    f"`identity_id` {idx} is out of range [0, {self.identity_bank_size - 1}]."
                )
        return ids

    def _append_identity_tokens(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_attention_mask: Optional[torch.Tensor],
        identity_id: Optional[Union[int, List[int], torch.Tensor]],
        identity_strength: float,
        identity_negative_strength: float,
        batch_size: int,
        num_videos_per_prompt: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.identity_bank_enabled:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            )
        if identity_id is None:
            return (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            )
        if identity_strength < 0:
            raise ValueError(f"`identity_strength` must be >= 0, got {identity_strength}.")
        if identity_negative_strength < 0:
            raise ValueError(
                f"`identity_negative_strength` must be >= 0, got {identity_negative_strength}."
            )

        base_ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)
        expanded_ids: List[int] = []
        for idx in base_ids:
            expanded_ids.extend([idx] * num_videos_per_prompt)

        bank_device = self.identity_embedding.weight.device
        id_index = torch.tensor(expanded_ids, dtype=torch.long, device=bank_device)
        id_tokens = self.identity_embedding(id_index).view(
            len(expanded_ids),
            self.identity_tokens_per_id,
            self.identity_token_dim,
        )
        id_tokens = id_tokens.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
        id_tokens = id_tokens.unsqueeze(1)  # [B, 1, N_id, D]

        pos_tokens = id_tokens * float(identity_strength)
        pos_mask = torch.ones(
            (pos_tokens.shape[0], pos_tokens.shape[2]),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        prompt_embeds = torch.cat([prompt_embeds, pos_tokens], dim=2)
        prompt_attention_mask = torch.cat([prompt_attention_mask, pos_mask], dim=1)

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is not None:
            neg_tokens = id_tokens.to(
                device=negative_prompt_embeds.device,
                dtype=negative_prompt_embeds.dtype,
            ) * float(identity_negative_strength)
            if identity_negative_strength > 0:
                neg_mask = torch.ones(
                    (neg_tokens.shape[0], neg_tokens.shape[2]),
                    dtype=negative_prompt_attention_mask.dtype,
                    device=negative_prompt_attention_mask.device,
                )
            else:
                neg_mask = torch.zeros(
                    (neg_tokens.shape[0], neg_tokens.shape[2]),
                    dtype=negative_prompt_attention_mask.dtype,
                    device=negative_prompt_attention_mask.device,
                )
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, neg_tokens], dim=2)
            negative_prompt_attention_mask = torch.cat([negative_prompt_attention_mask, neg_mask], dim=1)

        self.metrics.record("identity_tokens_appended", int(self.identity_tokens_per_id))
        self.metrics.record("identity_strength", float(identity_strength))
        self.metrics.record("identity_negative_strength", float(identity_negative_strength))
        self.metrics.record("identity_bank_active", 1)
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @torch.no_grad()
    def register_identity_from_latents(
        self,
        identity_id: Union[int, List[int], torch.Tensor],
        latents: torch.Tensor,
        momentum: float = 0.25,
    ) -> None:
        if not self.identity_bank_enabled:
            return
        if latents.ndim != 5:
            raise ValueError(f"`latents` must be [B, C, T, H, W], got shape {tuple(latents.shape)}.")
        if momentum < 0 or momentum > 1:
            raise ValueError(f"`momentum` must be in [0, 1], got {momentum}.")

        batch_size = latents.shape[0]
        ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)

        pooled = latents.to(torch.float32).mean(dim=(2, 3, 4))
        proj_device = next(self.identity_latent_projector.parameters()).device
        pooled = pooled.to(device=proj_device)
        projected = self.identity_latent_projector(pooled)  # [B, tokens*dim]

        with torch.no_grad():
            for b, idx in enumerate(ids):
                current = self.identity_embedding.weight[idx].detach().to(projected.dtype)
                observed = projected[b]
                updated = (1.0 - momentum) * current + momentum * observed
                cos = F.cosine_similarity(
                    current.unsqueeze(0),
                    observed.unsqueeze(0),
                    dim=-1,
                ).item()
                self.identity_embedding.weight[idx].data.copy_(
                    updated.to(self.identity_embedding.weight.dtype)
                )
                self.metrics.record("identity_bank_update_cosine", float(cos))
                self.metrics.record("identity_bank_updated_id", int(idx))

    @torch.no_grad()
    def save_identity_bank(self, path: str) -> str:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "version": 1,
            "timestamp": time.time(),
            "identity_bank_size": int(self.identity_bank_size),
            "identity_tokens_per_id": int(self.identity_tokens_per_id),
            "identity_token_dim": int(self.identity_token_dim),
            "identity_embedding": self.identity_embedding.weight.detach().cpu(),
            "identity_latent_projector": self.identity_latent_projector.state_dict(),
        }
        torch.save(payload, path)
        return path

    @torch.no_grad()
    def load_identity_bank(self, path: str, strict: bool = True) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        required_keys = {
            "version",
            "identity_bank_size",
            "identity_tokens_per_id",
            "identity_token_dim",
            "identity_embedding",
        }
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"Identity bank file is missing keys: {sorted(missing)}")

        loaded_bank = payload["identity_embedding"]
        if not isinstance(loaded_bank, torch.Tensor):
            raise ValueError("`identity_embedding` must be a torch.Tensor.")

        expected_shape = (
            self.identity_bank_size,
            self.identity_tokens_per_id * self.identity_token_dim,
        )
        loaded_shape = tuple(loaded_bank.shape)
        if strict and loaded_shape != expected_shape:
            raise ValueError(
                f"Identity bank shape mismatch. Expected {expected_shape}, got {loaded_shape}."
            )

        rows = min(expected_shape[0], loaded_bank.shape[0])
        cols = min(expected_shape[1], loaded_bank.shape[1])
        self.identity_embedding.weight.data[:rows, :cols].copy_(
            loaded_bank[:rows, :cols].to(self.identity_embedding.weight.dtype)
        )

        if "identity_latent_projector" in payload:
            self.identity_latent_projector.load_state_dict(payload["identity_latent_projector"], strict=False)

        self.metrics.record("identity_bank_loaded_rows", int(rows))
        self.metrics.record("identity_bank_loaded_cols", int(cols))
        return {
            "rows_loaded": int(rows),
            "cols_loaded": int(cols),
            "strict": bool(strict),
            "source": path,
        }

    @torch.no_grad()
    def enroll_identity_from_image(
        self,
        image: PipelineImageInput,
        identity_id: Union[int, List[int], torch.Tensor],
        resolution: Literal["480p", "720p"] = "480p",
        resize_mode: str = "crop",
        momentum: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Register (or update) one-shot identity slot(s) directly from reference image(s)
        without running a full diffusion sampling loop.
        """
        if resize_mode not in ("default", "crop"):
            raise ValueError(f"Unsupported resize_mode {resize_mode}, and you can only choose from [default, crop]")
        if identity_id is None:
            raise ValueError("`identity_id` is required for identity enrollment.")

        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)

        height, width = self.get_condition_shape(
            image,
            resolution,
            scale_factor_spatial=scale_factor_spatial,
        )
        image_tensor = self.video_processor.preprocess(
            image,
            height=height,
            width=width,
            resize_mode=resize_mode,
        ).to(device=self.device, dtype=self.dit.dtype)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        batch_size = image_tensor.shape[0]
        latents = self.prepare_latents(
            image=image_tensor,
            batch_size=batch_size,
            num_channels_latents=self.dit.config.in_channels,
            height=height,
            width=width,
            num_frames=1,
            num_cond_frames=1,
            dtype=torch.float32,
            device=self.device,
            generator=None,
            latents=None,
        )
        cond_latents = latents[:, :, :1]
        self.register_identity_from_latents(
            identity_id=identity_id,
            latents=cond_latents,
            momentum=momentum,
        )
        normalized_ids = self._normalize_identity_ids(identity_id, batch_size=batch_size)
        self.metrics.record("identity_enroll_batch_size", int(batch_size))
        self.metrics.record("identity_enroll_momentum", float(momentum))
        return {
            "identity_ids": normalized_ids,
            "batch_size": int(batch_size),
            "height": int(height),
            "width": int(width),
            "momentum": float(momentum),
        }

    def _compress_kv_pair_temporal(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        num_cond_latents: int,
        num_ref_latents: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compress KV cache along temporal dimension by preserving:
        1) reference latents (if present),
        2) a summarized memory of old frames,
        3) a recent exact sliding window.
        """
        if not self.temporal_memory_enabled:
            return k.contiguous(), v.contiguous(), num_cond_latents

        if num_cond_latents <= 0 or k.ndim != 4 or v.ndim != 4:
            return k.contiguous(), v.contiguous(), num_cond_latents

        if k.shape != v.shape:
            return k.contiguous(), v.contiguous(), num_cond_latents

        seq_len = k.shape[2]
        if seq_len % num_cond_latents != 0:
            # Keep cache untouched if token layout is unknown.
            return k.contiguous(), v.contiguous(), num_cond_latents

        tokens_per_frame = seq_len // num_cond_latents
        preserve_ref_frames = max(0, min(num_ref_latents or 0, num_cond_latents))
        non_ref_frames = num_cond_latents - preserve_ref_frames

        # Nothing to compress.
        if non_ref_frames <= 0:
            return k.contiguous(), v.contiguous(), num_cond_latents

        keep_recent = max(1, int(self.temporal_memory_window_frames))
        keep_recent = min(keep_recent, non_ref_frames)
        old_frames = non_ref_frames - keep_recent

        if old_frames <= 0:
            return k.contiguous(), v.contiguous(), num_cond_latents

        summary_frames = max(0, int(self.temporal_memory_summary_frames))
        summary_frames = min(summary_frames, old_frames)
        if summary_frames <= 0:
            summary_frames = 1

        B, Hh, _, Dd = k.shape
        ref_tokens = preserve_ref_frames * tokens_per_frame
        old_tokens = old_frames * tokens_per_frame

        ref_k = k[:, :, :ref_tokens, :] if ref_tokens > 0 else None
        ref_v = v[:, :, :ref_tokens, :] if ref_tokens > 0 else None

        old_k = k[:, :, ref_tokens:ref_tokens + old_tokens, :]
        old_v = v[:, :, ref_tokens:ref_tokens + old_tokens, :]
        recent_k = k[:, :, ref_tokens + old_tokens:, :]
        recent_v = v[:, :, ref_tokens + old_tokens:, :]

        old_k = old_k.view(B, Hh, old_frames, tokens_per_frame, Dd)
        old_v = old_v.view(B, Hh, old_frames, tokens_per_frame, Dd)

        k_summaries = []
        v_summaries = []
        for i in range(summary_frames):
            start = (i * old_frames) // summary_frames
            end = ((i + 1) * old_frames) // summary_frames
            if end <= start:
                end = min(old_frames, start + 1)
            if end <= start:
                continue
            k_summaries.append(old_k[:, :, start:end, :, :].mean(dim=2, keepdim=True))
            v_summaries.append(old_v[:, :, start:end, :, :].mean(dim=2, keepdim=True))

        if k_summaries:
            old_k_summary = torch.cat(k_summaries, dim=2).reshape(B, Hh, -1, Dd)
            old_v_summary = torch.cat(v_summaries, dim=2).reshape(B, Hh, -1, Dd)
            summary_frames_eff = old_k_summary.shape[2] // tokens_per_frame
        else:
            old_k_summary = k.new_empty(B, Hh, 0, Dd)
            old_v_summary = v.new_empty(B, Hh, 0, Dd)
            summary_frames_eff = 0

        parts_k = []
        parts_v = []
        if ref_k is not None:
            parts_k.append(ref_k)
            parts_v.append(ref_v)
        parts_k.append(old_k_summary)
        parts_v.append(old_v_summary)
        parts_k.append(recent_k)
        parts_v.append(recent_v)

        k_comp = torch.cat(parts_k, dim=2).contiguous()
        v_comp = torch.cat(parts_v, dim=2).contiguous()
        effective_cond_latents = preserve_ref_frames + summary_frames_eff + keep_recent

        return k_comp, v_comp, effective_cond_latents

    def _compress_kv_cache_dict_temporal(
        self,
        kv_cache_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        num_cond_latents: int,
        num_ref_latents: int,
    ) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], int]:
        if not self.temporal_memory_enabled or not kv_cache_dict:
            return kv_cache_dict, num_cond_latents

        compressed: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        effective_latent_counts: List[int] = []

        for layer_idx, cache in kv_cache_dict.items():
            if not isinstance(cache, tuple) or len(cache) != 2:
                compressed[layer_idx] = cache
                continue
            k, v = cache
            k_comp, v_comp, eff_count = self._compress_kv_pair_temporal(
                k,
                v,
                num_cond_latents=num_cond_latents,
                num_ref_latents=num_ref_latents,
            )
            compressed[layer_idx] = (k_comp, v_comp)
            effective_latent_counts.append(eff_count)

        if not effective_latent_counts:
            return compressed, num_cond_latents

        effective_num_cond_latents = min(effective_latent_counts)
        return compressed, effective_num_cond_latents

    def _cache_clean_latents(self, cond_latents, model_max_length, offload_kv_cache, device, dtype, audio_embs, num_cond_latents, num_ref_latents, ref_img_index):
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2]).to(device=device, dtype=dtype)
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        empty_embeds = torch.zeros([cond_latents.shape[0], 1, model_max_length, self.text_encoder.config.d_model], device=device, dtype=dtype)
        _, kv_cache_dict = self.dit(
            hidden_states=cond_latents, 
            timestep=timestep, 
            encoder_hidden_states=empty_embeds,
            num_cond_latents=num_cond_latents,
            return_kv=True, 
            skip_crs_attn=True, 
            offload_kv_cache=offload_kv_cache,
            audio_embs=audio_embs,
            num_ref_latents=num_ref_latents,
            ref_img_index=ref_img_index
        )
        effective_num_cond_latents = num_cond_latents
        if self.temporal_memory_enabled:
            kv_cache_dict, effective_num_cond_latents = self._compress_kv_cache_dict_temporal(
                kv_cache_dict,
                num_cond_latents=num_cond_latents,
                num_ref_latents=num_ref_latents or 0,
            )
            if effective_num_cond_latents != num_cond_latents:
                self.metrics.record("kv_cache_cond_latents_before", num_cond_latents)
                self.metrics.record("kv_cache_cond_latents_after", effective_num_cond_latents)
        self._update_kv_cache_dict(kv_cache_dict)
        return effective_num_cond_latents
    
    def _get_kv_cache_dict(self):
        return self.kv_cache_dict
    
    def _clear_cache(self):
        self.kv_cache_dict = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_condition_shape(self, condition, resolution, scale_factor_spatial=32):
        bucket_config = get_bucket_config(resolution, scale_factor_spatial=scale_factor_spatial)

        obj = condition[0] if isinstance(condition, list) and condition else condition
        try:
            height = getattr(obj, "height")
            width = getattr(obj, "width")
        except AttributeError:
            raise ValueError("Unsupported condition type")

        ratio = height / width
        # Find the closest bucket
        closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        return target_h, target_w
    
    def optimized_scale(self, positive_flat, negative_flat):
        """ from CFG-zero paper
        """
        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        # st_star = v_condˆT * v_uncond / ||v_uncond||ˆ2
        st_star = dot_product / squared_norm
        return st_star
    
    def normalize_latents(self, latents):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return (latents - latents_mean) * latents_std

    def denormalize_latents(self, latents):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return latents / latents_std + latents_mean

    def _loudness_norm(self, audio_array, sr=16000, lufs=-23, threshold=100):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > threshold:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def _add_noise_floor(self, audio, noise_db=-45):
        noise_amp = 10 ** (noise_db / 20)
        noise = np.random.randn(len(audio)) * noise_amp
        return audio + noise

    def _smooth_transients(self, audio, sr=16000):
        b, a = ss.butter(3, 3000 / (sr/2))
        return ss.lfilter(b, a, audio)
    
    def _resize_and_centercrop_tensor(self, mask: torch.Tensor, target_h: int, target_w: int, resize_mode: str = 'crop'):
        """
        mask: Tensor, shape [3, H, W], dtype=float, device=gpu/cpu
        return: [3, target_h, target_w]
        """

        if resize_mode == 'default':
            mask_resized = F.interpolate(
                mask.unsqueeze(0),  # [1, 3, H, W]
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            return mask_resized

        elif resize_mode == 'crop':
            _, H, W = mask.shape
            ratio = target_w / target_h # 1
            src_ratio = W / H # > 1

            if ratio > src_ratio:
                new_w = target_w
                new_h = int(H * target_w / W)
            else:
                new_h = target_h
                new_w = int(W * target_h / H)

            mask_resized = F.interpolate(
                mask.unsqueeze(0),  # [1, 3, H, W]
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            top = (new_h - target_h) // 2
            left = (new_w - target_w) // 2

            mask_resized_cropped = mask_resized[:, top:top + target_h, left:left + target_w]
            return mask_resized_cropped
        
        else:
            raise ValueError(f"Unsupported resize_mode {resize_mode}. Use 'default' or 'crop'.")

    def _apply_multistream_fusion(
        self,
        audio_emb: torch.Tensor,
        fused_emb: Optional[torch.Tensor],
        device: torch.device
    ) -> torch.Tensor:
        if fused_emb is None:
            return audio_emb
        if audio_emb.dim() == 3:
            audio_bt = audio_emb.permute(1, 0, 2).contiguous()
        elif audio_emb.dim() == 2:
            audio_bt = audio_emb.unsqueeze(0)
        else:
            return audio_emb

        fused = fused_emb.to(device=device, dtype=audio_bt.dtype)
        proj = self.multi_stream_fusion_proj.to(device=device, dtype=audio_bt.dtype)
        fused_proj = proj(fused)  # [B, min_t, 768]
        fused_proj = fused_proj.permute(0, 2, 1)
        fused_proj = F.interpolate(
            fused_proj, size=audio_bt.shape[1], mode="linear", align_corners=False
        )
        fused_proj = fused_proj.permute(0, 2, 1)
        audio_bt = audio_bt + self.multi_stream_fusion_scale * fused_proj

        if audio_emb.dim() == 3:
            return audio_bt.permute(1, 0, 2).contiguous()
        return audio_bt.squeeze(0)

    @torch.no_grad()
    def _extract_phoneme_timeline(
        self,
        speech_array: np.ndarray,
        sample_rate: int,
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, Any]]:
        if not self.phoneme_enabled or self.phoneme_aligner is None or target_len <= 0:
            return None

        try:
            with self.metrics.timeit("phoneme_extract"):
                phoneme_out = self.phoneme_aligner.extract(
                    speech_array,
                    sample_rate=sample_rate,
                    target_len=target_len,
                )

            phoneme_probs = phoneme_out["phoneme_probs"].to(device=device, dtype=dtype)
            phoneme_ids = phoneme_out["phoneme_ids"].to(device=device, dtype=torch.long)
            confidence = phoneme_out["confidence"].to(device=device, dtype=dtype)

            self.metrics.record("phoneme_voiced_ratio", float(phoneme_out.get("voiced_ratio", 0.0)))
            self.metrics.record("phoneme_silence_ratio", float(phoneme_out.get("silence_ratio", 0.0)))
            self.metrics.record("phoneme_fricative_ratio", float(phoneme_out.get("fricative_ratio", 0.0)))
            self.metrics.record("phoneme_plosive_ratio", float(phoneme_out.get("plosive_ratio", 0.0)))
            self.metrics.record("phoneme_confidence_mean", float(confidence.mean().item()))

            return {
                "phoneme_probs": phoneme_probs,
                "phoneme_ids": phoneme_ids,
                "confidence": confidence,
            }
        except Exception as exc:
            self.metrics.record("phoneme_fallback_count", 1)
            if not self.phoneme_fallback_to_wav2vec:
                raise
            loguru.logger.warning(
                "Phoneme extraction failed; using wav2vec fallback only. Error: {}",
                exc,
            )
            return None

    def _inject_phoneme_conditioning(
        self,
        audio_emb: torch.Tensor,
        phoneme_probs: torch.Tensor,
        confidence: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if audio_emb.dim() not in (2, 3):
            return audio_emb

        phoneme_proj = self.phoneme_proj.to(device=device, dtype=audio_emb.dtype)
        phoneme_ctx = phoneme_proj(phoneme_probs.to(device=device, dtype=audio_emb.dtype))
        conf = confidence.to(device=device, dtype=audio_emb.dtype).clamp(
            min=float(self.phoneme_confidence_floor), max=1.0
        )
        phoneme_ctx = phoneme_ctx * conf.unsqueeze(-1)

        if audio_emb.dim() == 3:
            phoneme_ctx = phoneme_ctx.unsqueeze(1).expand(-1, audio_emb.shape[1], -1)

        conditioned = audio_emb + float(self.phoneme_stream_scale) * phoneme_ctx
        return conditioned.contiguous()

    def _compute_phoneme_alignment_metrics(
        self,
        audio_emb: torch.Tensor,
        phoneme_probs: torch.Tensor,
        phoneme_ids: torch.Tensor,
        device: torch.device,
    ) -> Optional[Dict[str, float]]:
        if audio_emb.dim() == 3:
            frame_repr = audio_emb.mean(dim=1)
        elif audio_emb.dim() == 2:
            frame_repr = audio_emb
        else:
            return None

        target_len = phoneme_probs.shape[0]
        if target_len <= 0:
            return None

        if frame_repr.shape[0] != target_len:
            frame_repr = frame_repr.transpose(0, 1).unsqueeze(0)
            frame_repr = F.interpolate(frame_repr, size=target_len, mode="linear", align_corners=False)
            frame_repr = frame_repr.squeeze(0).transpose(0, 1).contiguous()

        frame_repr = frame_repr.to(device=device, dtype=torch.float32)
        probs = phoneme_probs.to(device=device, dtype=torch.float32)
        ids = phoneme_ids.to(device=device, dtype=torch.long)

        head = self.phoneme_alignment_head.to(device=device, dtype=frame_repr.dtype)
        logits = head(frame_repr)
        log_probs = F.log_softmax(logits, dim=-1)

        kl = F.kl_div(log_probs, probs, reduction="batchmean")
        ce = F.nll_loss(log_probs, ids, reduction="mean")
        loss = 0.5 * (kl + ce)
        pred = torch.argmax(log_probs, dim=-1)
        acc = (pred == ids).float().mean()

        return {
            "loss": float(loss.item()),
            "kl": float(kl.item()),
            "ce": float(ce.item()),
            "acc": float(acc.item()),
        }

    def _normalize_emotion_ids(
        self,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]],
        batch_size: int,
    ) -> Optional[List[int]]:
        if emotion_id is None:
            return None

        if isinstance(emotion_id, (int, str)):
            raw_items: List[Union[int, str]] = [emotion_id] * batch_size
        elif isinstance(emotion_id, torch.Tensor):
            raw_items = [int(x) for x in emotion_id.detach().cpu().view(-1).tolist()]
        elif isinstance(emotion_id, (list, tuple)):
            raw_items = list(emotion_id)
        else:
            raise TypeError(
                f"`emotion_id` must be int, str, list, torch.Tensor, or None. Got {type(emotion_id)}."
            )

        if len(raw_items) == 1 and batch_size > 1:
            raw_items = raw_items * batch_size
        if len(raw_items) != batch_size:
            raise ValueError(
                f"`emotion_id` length must be 1 or equal to batch size ({batch_size}), got {len(raw_items)}."
            )

        resolved: List[int] = []
        for item in raw_items:
            if isinstance(item, str):
                key = item.strip().lower()
                if key not in self.emotion_label_to_id:
                    raise ValueError(
                        f"Unknown emotion label `{item}`. Allowed: {sorted(self.emotion_label_to_id.keys())}."
                    )
                resolved.append(int(self.emotion_label_to_id[key]))
            else:
                idx = int(item)
                if idx < 0 or idx >= self.emotion_num_classes:
                    raise ValueError(
                        f"`emotion_id` {idx} is out of range [0, {self.emotion_num_classes - 1}]."
                    )
                resolved.append(idx)
        return resolved

    def _apply_emotion_channel(
        self,
        audio_emb: torch.Tensor,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]],
        emotion_intensity: float,
        batch_size: int,
        num_videos_per_prompt: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, bool]:
        if not self.emotion_enabled or emotion_id is None:
            return audio_emb, False
        if emotion_intensity <= 0:
            return audio_emb, False
        if audio_emb.ndim != 5:
            return audio_emb, False

        ids = self._normalize_emotion_ids(emotion_id, batch_size=batch_size)
        expanded_ids: List[int] = []
        for idx in ids:
            expanded_ids.extend([idx] * num_videos_per_prompt)

        emb = self.emotion_embedding(
            torch.tensor(expanded_ids, dtype=torch.long, device=self.emotion_embedding.weight.device)
        ).to(device=device, dtype=audio_emb.dtype)
        emb = self.emotion_proj.to(device=device, dtype=audio_emb.dtype)(emb)

        B = audio_emb.shape[0]
        emotion_ctx = emb.view(B, 1, 1, 1, -1)

        audio_rms = (
            audio_emb.to(torch.float32).pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
        )
        emotion_rms = (
            emotion_ctx.to(torch.float32).pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().clamp_min(1e-6)
        )
        requested_scale = torch.full_like(audio_rms, float(emotion_intensity))
        max_scale = (audio_rms * float(self.emotion_lipsync_guard_ratio)) / emotion_rms
        safe_scale = torch.minimum(requested_scale, max_scale)
        safe_scale = torch.clamp(safe_scale, min=0.0)

        conditioned = audio_emb + emotion_ctx * safe_scale.to(device=device, dtype=audio_emb.dtype)

        requested = float(emotion_intensity)
        applied = float(safe_scale.mean().item())
        clipped = bool(applied + 1e-6 < requested)
        self.metrics.record("emotion_intensity_requested", requested)
        self.metrics.record("emotion_intensity_applied", applied)
        self.metrics.record("emotion_lipsync_guard_triggered", int(clipped))
        self.metrics.record("emotion_lipsync_guard_ratio", float(self.emotion_lipsync_guard_ratio))

        return conditioned.contiguous(), applied > 0.0

    def _prepare_mouth_zone_mask(
        self,
        mouth_zone_masks: Optional[torch.Tensor],
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        resize_mode: str = "crop",
    ) -> Optional[torch.Tensor]:
        if mouth_zone_masks is None:
            return None

        mask = mouth_zone_masks
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        mask = mask.to(device=device, dtype=torch.float32)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif mask.ndim == 3:
            if mask.shape[0] in (1, 3):
                mask = mask.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
            else:
                mask = mask.unsqueeze(1)  # [B,1,H,W]
        elif mask.ndim == 4:
            if mask.shape[1] in (1, 3):
                mask = mask.mean(dim=1, keepdim=True)  # [B,1,H,W]
            else:
                return None
        else:
            return None

        if mask.shape[-2:] != (height, width):
            if mask.shape[0] == 1:
                m3 = mask[0].repeat(3, 1, 1)
                m3 = self._resize_and_centercrop_tensor(m3, height, width, resize_mode)
                mask = m3.mean(dim=0, keepdim=True).unsqueeze(0)
            else:
                resized = []
                for b in range(mask.shape[0]):
                    m3 = mask[b].repeat(3, 1, 1)
                    m3 = self._resize_and_centercrop_tensor(m3, height, width, resize_mode)
                    resized.append(m3.mean(dim=0, keepdim=True))
                mask = torch.stack(resized, dim=0)

        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1).contiguous()
        elif mask.shape[0] != batch_size:
            return None

        mask = torch.clamp(mask, 0.0, 1.0)
        for _ in range(max(1, int(self.hybrid_renderer_blur_passes))):
            mask = F.avg_pool2d(mask, kernel_size=5, stride=1, padding=2)
        mask = torch.clamp(mask, 0.0, 1.0).to(dtype=dtype)
        mask = mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)  # [B,1,T,H,W]
        return mask.contiguous()

    def _compute_seam_boundary_mask(self, mouth_mask: torch.Tensor) -> torch.Tensor:
        # High value on transition ring around mouth mask.
        b, _, t, h, w = mouth_mask.shape
        flat = mouth_mask.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        eroded = F.avg_pool2d(flat, kernel_size=5, stride=1, padding=2)
        ring = torch.clamp(flat - eroded, min=0.0, max=1.0)
        ring = torch.clamp(ring * 4.0, min=0.0, max=1.0)
        ring = ring.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return ring

    def _build_mouth_controlled_branch(self, decoded_video: torch.Tensor, strength: float) -> torch.Tensor:
        # Deterministic high-frequency enhancement branch for the controlled zone.
        b, c, t, h, w = decoded_video.shape
        flat = decoded_video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        blurred = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1)
        detail = flat - blurred
        branch = flat + float(strength) * detail
        return branch.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def _temporal_stabilize_boundary(
        self,
        hybrid_video: torch.Tensor,
        boundary_mask: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        if hybrid_video.shape[2] <= 1:
            return hybrid_video

        stabilized = hybrid_video.clone()
        a = float(alpha)
        for i in range(1, stabilized.shape[2]):
            prev = stabilized[:, :, i - 1]
            curr = stabilized[:, :, i]
            seam = boundary_mask[:, :, i]
            blended = a * curr + (1.0 - a) * prev
            stabilized[:, :, i] = curr * (1.0 - seam) + blended * seam
        return stabilized

    def _validate_hybrid_renderer_budget(
        self,
        global_video: torch.Tensor,
        hybrid_video: torch.Tensor,
        mouth_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> None:
        if hybrid_video.shape[2] <= 1:
            return

        g = global_video.to(torch.float32)
        h = hybrid_video.to(torch.float32)

        dt_g = torch.abs(g[:, :, 1:] - g[:, :, :-1])
        dt_h = torch.abs(h[:, :, 1:] - h[:, :, :-1])
        seam = boundary_mask[:, :, 1:]
        seam_den = seam.mean().clamp_min(1e-6)
        flicker_g = (dt_g * seam).mean() / seam_den
        flicker_h = (dt_h * seam).mean() / seam_den
        flicker_ratio = (flicker_h / flicker_g.clamp_min(1e-6)).item()

        artifact_energy = (torch.abs(h - g) * mouth_mask).mean().item()
        self.metrics.record("hybrid_mouth_flicker_ratio", float(flicker_ratio))
        self.metrics.record("hybrid_mouth_artifact_energy", float(artifact_energy))
        self.metrics.record("hybrid_mouth_budget_ok", int(
            flicker_ratio <= float(self.hybrid_renderer_flicker_budget)
            and artifact_energy <= float(self.hybrid_renderer_artifact_budget)
        ))

        if flicker_ratio > float(self.hybrid_renderer_flicker_budget):
            loguru.logger.warning(
                "Hybrid mouth renderer flicker budget exceeded: ratio {:.4f} > {:.4f}",
                flicker_ratio,
                float(self.hybrid_renderer_flicker_budget),
            )
        if artifact_energy > float(self.hybrid_renderer_artifact_budget):
            loguru.logger.warning(
                "Hybrid mouth renderer artifact budget exceeded: {:.6f} > {:.6f}",
                artifact_energy,
                float(self.hybrid_renderer_artifact_budget),
            )

    def _apply_hybrid_mouth_renderer(
        self,
        decoded_video: torch.Tensor,
        mouth_zone_masks: Optional[torch.Tensor],
        resize_mode: str = "crop",
    ) -> torch.Tensor:
        if not self.hybrid_renderer_enabled or mouth_zone_masks is None:
            return decoded_video
        if decoded_video.ndim != 5:
            return decoded_video

        b, _, t, h, w = decoded_video.shape
        mouth_mask = self._prepare_mouth_zone_mask(
            mouth_zone_masks=mouth_zone_masks,
            batch_size=b,
            num_frames=t,
            height=h,
            width=w,
            device=decoded_video.device,
            dtype=decoded_video.dtype,
            resize_mode=resize_mode,
        )
        if mouth_mask is None:
            return decoded_video

        boundary_mask = self._compute_seam_boundary_mask(mouth_mask)
        mouth_branch = self._build_mouth_controlled_branch(
            decoded_video=decoded_video,
            strength=float(self.hybrid_renderer_mouth_strength),
        )
        hybrid = decoded_video * (1.0 - mouth_mask) + mouth_branch * mouth_mask
        hybrid = self._temporal_stabilize_boundary(
            hybrid_video=hybrid,
            boundary_mask=boundary_mask,
            alpha=float(self.hybrid_renderer_temporal_alpha),
        )
        self._validate_hybrid_renderer_budget(
            global_video=decoded_video,
            hybrid_video=hybrid,
            mouth_mask=mouth_mask,
            boundary_mask=boundary_mask,
        )
        return hybrid.contiguous()

    @torch.no_grad()
    def get_audio_embedding(self, speech_array, fps=32, device='cpu', sample_rate=16000):
            
        # optional disk cache for audio embeddings to accelerate repeated runs
        cache_dir = getattr(self, 'audio_cache_dir', './audio_cache')
        os.makedirs(cache_dir, exist_ok=True)

        phoneme_scale_tag = str(round(float(self.phoneme_stream_scale), 4)).replace(".", "p")
        key = (
            sha256_of_audio_array(np.ascontiguousarray(speech_array))
            + f"_fps{fps}_sr{sample_rate}_ph{int(bool(self.phoneme_enabled))}_pn{self.phoneme_num_classes}_ps{phoneme_scale_tag}_v2"
        )
        cache_path = os.path.join(cache_dir, key + '.npz')

        if os.path.exists(cache_path):
            try:
                with self.metrics.timeit('audio_cache_load'):
                    npz = np.load(cache_path)
                    if "audio_emb_final" in npz:
                        cached = npz["audio_emb_final"]
                    else:
                        cached = npz["audio_emb"]
                    audio_emb = torch.from_numpy(cached).to(device=device)
                    # return shape (T, B, D)
                    return audio_emb
            except Exception as exc:
                loguru.logger.debug("Audio cache load failed; recomputing. Error: {}", exc)

        audio_duration = len(speech_array) / sample_rate
        video_length = audio_duration * fps

        # speech preprocess
        speech_array = self._loudness_norm(speech_array, sample_rate)
        speech_array = self._add_noise_floor(speech_array)
        speech_array = self._smooth_transients(speech_array)

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sample_rate).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)

        # audio embedding
        with self.metrics.timeit('wav2vec_encode'):
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d").contiguous() # T, 12, 768


        # try to compute fused multi-stream features and persist to cache
        try:
            fused_emb = None
            try:
                # audio_emb shape may be [T, B, D] or [T, D]
                a = audio_emb
                if a.dim() == 3:
                    # [T, B, D] -> [B, T, D]
                    wav2vec_feats = a.permute(1, 0, 2).contiguous()
                elif a.dim() == 2:
                    wav2vec_feats = a.unsqueeze(0)
                else:
                    wav2vec_feats = a

                processor_in = wav2vec_feats.cpu()
                proc_out = self.audio_processor(processor_in)
                fused_emb = proc_out.get('fused_embeddings', None)
            except Exception as exc:
                loguru.logger.debug(
                    "Audio multi-stream processor step failed; fused embeddings disabled. Error: {}",
                    exc,
                )
                fused_emb = None

            if fused_emb is not None:
                audio_emb = self._apply_multistream_fusion(audio_emb, fused_emb, device=device)
        except Exception as exc:
            loguru.logger.debug("Multi-stream fusion failed; continuing with wav2vec stream only. Error: {}", exc)

        phoneme_ctx = self._extract_phoneme_timeline(
            speech_array=speech_array,
            sample_rate=sample_rate,
            target_len=audio_emb.shape[0],
            device=device,
            dtype=audio_emb.dtype,
        )
        if phoneme_ctx is not None:
            audio_emb = self._inject_phoneme_conditioning(
                audio_emb=audio_emb,
                phoneme_probs=phoneme_ctx["phoneme_probs"],
                confidence=phoneme_ctx["confidence"],
                device=device,
            )
            align_metrics = self._compute_phoneme_alignment_metrics(
                audio_emb=audio_emb,
                phoneme_probs=phoneme_ctx["phoneme_probs"],
                phoneme_ids=phoneme_ctx["phoneme_ids"],
                device=device,
            )
            if align_metrics is not None:
                self.metrics.record("phoneme_alignment_loss", align_metrics["loss"])
                self.metrics.record("phoneme_alignment_kl", align_metrics["kl"])
                self.metrics.record("phoneme_alignment_ce", align_metrics["ce"])
                self.metrics.record("phoneme_alignment_acc", align_metrics["acc"])

        try:
            payload = {
                "audio_emb": audio_emb.cpu().numpy(),
                "audio_emb_final": audio_emb.cpu().numpy(),
            }
            if phoneme_ctx is not None:
                payload["phoneme_probs"] = phoneme_ctx["phoneme_probs"].cpu().numpy()
                payload["phoneme_confidence"] = phoneme_ctx["confidence"].cpu().numpy()
            np.savez_compressed(cache_path, **payload)
            self.metrics.record('audio_cache_saved', 1)
        except Exception as exc:
            loguru.logger.warning("Audio cache save failed; continuing without cache. Error: {}", exc)

        return audio_emb

    def _build_windowed_audio_embedding(
        self,
        full_audio_emb: torch.Tensor,
        num_frames: int,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        if full_audio_emb.dim() == 2:
            full_audio_emb = full_audio_emb.unsqueeze(1)
        if full_audio_emb.dim() != 3:
            raise ValueError(
                f"Expected full audio embedding with shape [T, S, C], got {tuple(full_audio_emb.shape)}."
            )
        if full_audio_emb.shape[0] <= 0:
            raise ValueError("Audio embedding has no timesteps.")

        audio_window = int(getattr(self.dit, "audio_window", 5))
        audio_window = max(1, 2 * (audio_window // 2) + 1)
        audio_stride = max(int(self.vae_scale_factor_temporal), 1)

        offsets = torch.arange(audio_window, device=full_audio_emb.device) - (audio_window // 2)
        center_indices = torch.arange(
            0,
            audio_stride * int(num_frames),
            audio_stride,
            device=full_audio_emb.device,
        ).unsqueeze(1) + offsets.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)

        windowed = full_audio_emb[center_indices][None, ...]  # [1, T, W, S, C]
        return windowed.to(device=device, dtype=self.dit.dtype)

    def _prepare_audio_emb_for_dit(
        self,
        audio_emb: torch.Tensor,
        *,
        num_frames: int,
        batch_size: int,
        num_videos_per_prompt: int,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        if audio_emb is None:
            raise ValueError("`audio_emb` is required for audio-driven generation.")

        if not torch.is_tensor(audio_emb):
            audio_emb = torch.as_tensor(audio_emb)

        if audio_emb.dim() == 3:
            audio_emb = self._build_windowed_audio_embedding(audio_emb, num_frames=num_frames, device=device)
        elif audio_emb.dim() == 4:
            audio_emb = audio_emb.unsqueeze(0)
        elif audio_emb.dim() != 5:
            raise ValueError(
                f"`audio_emb` must be 3D [T,S,C], 4D [T,W,S,C], or 5D [B,T,W,S,C], got {tuple(audio_emb.shape)}."
            )

        expected_time = int(num_frames)
        if audio_emb.shape[1] != expected_time:
            raise ValueError(
                f"`audio_emb` time dimension mismatch: expected {expected_time}, got {audio_emb.shape[1]}."
            )

        expected_window = max(1, 2 * (int(getattr(self.dit, "audio_window", 5)) // 2) + 1)
        if audio_emb.shape[2] != expected_window:
            raise ValueError(
                f"`audio_emb` window dimension mismatch: expected {expected_window}, got {audio_emb.shape[2]}."
            )

        target_batch = int(batch_size) * int(num_videos_per_prompt)
        source_batch = int(audio_emb.shape[0])
        if source_batch == 1 and target_batch > 1:
            audio_emb = audio_emb.expand(target_batch, -1, -1, -1, -1).contiguous()
        elif source_batch == int(batch_size) and int(num_videos_per_prompt) > 1:
            audio_emb = audio_emb.repeat_interleave(int(num_videos_per_prompt), dim=0)
        elif source_batch != target_batch:
            raise ValueError(
                f"`audio_emb` batch dimension mismatch: expected 1, {batch_size}, or {target_batch}, got {source_batch}."
            )

        return audio_emb.to(device=device, dtype=self.dit.dtype, non_blocking=True)

    @torch.no_grad()
    def generate_at2v(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # avatar related params
        audio_emb: torch.Tensor = None,
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
        resize_mode: Optional[str] = "crop",
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            prompt (`str or List[str]`):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            height (`int`, *optional*, defaults to 480):
                Height of each video frame. Must be divisible by 16.
            width (`int`, *optional*, defaults to 832):
                Width of each video frame. Must be divisible by 16.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            identity_id (`int` or `List[int]`, *optional*):
                Identity slot index (or per-sample indices) in the learnable identity token bank.
            identity_strength (`float`, *optional*, defaults to 1.0):
                Scale applied to identity tokens for conditioned branch.
            identity_negative_strength (`float`, *optional*, defaults to 0.0):
                Scale applied to identity tokens for unconditioned branch.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if emotion_guidance_scale > 0 and (emotion_id is None or emotion_intensity <= 0):
            loguru.logger.warning(
                "Emotion guidance is enabled but emotion control is missing; disabling emotion guidance for this call."
            )
            emotion_guidance_scale = 0.0

        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._emotion_guidance_scale = float(emotion_guidance_scale)
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)


        # 3. Encode inputs
        dit_dtype = self.dit.dtype
        identity_token_count = (
            self.identity_tokens_per_id
            if self.identity_bank_enabled and identity_id is not None
            else 0
        )

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds, 
                prompt_attention_mask, 
                negative_prompt_embeds, 
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._append_identity_tokens(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                identity_id=identity_id,
                identity_strength=identity_strength,
                identity_negative_strength=identity_negative_strength,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_seq_len = max_sequence_length + identity_token_count
            effective_batch_size = batch_size * num_videos_per_prompt
            prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_base_embs = self._prepare_audio_emb_for_dit(
            audio_emb,
            num_frames=num_frames,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_cond_embs, emotion_active = self._apply_emotion_channel(
            audio_emb=audio_base_embs,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_guidance_embs = None
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_base_embs)
            if emotion_active and self.emotion_guidance_scale > 0.0:
                audio_guidance_embs = audio_base_embs
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)

        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.dit.config.in_channels
            
        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        # 6. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        start_time = time.time()
        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    raise GenerationInterrupted()

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)

                noise_pred_cond = self._predict_avatar_noise(
                    hidden_states=latents,
                    timestep=timestep[: latents.shape[0]],
                    encoder_hidden_states=prompt_embeds[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_attention_mask,
                    audio_embs=audio_cond_embs[latents.shape[0] :] if self.do_classifier_free_guidance else audio_cond_embs,
                )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    noise_pred_uncond = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        audio_embs=audio_unond_embs,
                    )
                    noise_pred_text = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                        encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                        audio_embs=audio_unond_embs,
                    )

                    if emotion_active and self.emotion_guidance_scale > 0.0 and audio_guidance_embs is not None:
                        noise_pred_audio = self._predict_avatar_noise(
                            hidden_states=latents,
                            timestep=timestep_uncond,
                            encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                            encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                            audio_embs=audio_guidance_embs,
                        )
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_audio - noise_pred_text)
                            + self.emotion_guidance_scale * (noise_pred_cond - noise_pred_audio)
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_cond - noise_pred_text)
                        )
                else:
                    noise_pred = noise_pred_cond

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        total_time = time.time() - start_time
        try:
            self.metrics.record('denoise_seconds', total_time)
            self.metrics.record('denoise_p95', total_time)
        except Exception as exc:
            loguru.logger.debug("Metric logging failed; continuing. Error: {}", exc)

        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self.vae.decode(latents, return_dict=False)[0]
        output_video = self._apply_hybrid_mouth_renderer(
            decoded_video=output_video,
            mouth_zone_masks=mouth_zone_masks,
            resize_mode=resize_mode,
        )
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else:
            return output_video
    

    @torch.no_grad()
    def generate_ai2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        # avatar related params
        audio_emb: torch.Tensor = None,
        ref_target_masks: torch.Tensor = None,
        resize_mode: Optional[str] = "crop", # "default" / "crop"
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        update_identity_bank: bool = False,
        identity_update_momentum: float = 0.25,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
    ):
        r"""
        Generates video frames from an input image and text prompt using diffusion process.

        Args:
            image (`PipelineImageInput`):
                Input image for video generation.
            prompt (`str or List[str]`, *optional*):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            resolution (`Literal["480p", "720p"]`, *optional*, defaults to "480p"):
                Target video resolution. Determines output frame size.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            ref_target_masks(`torch.Tensor`, *optional*, defaults to None):
                Mask used in dual-speaker audio-driven mode.
            resize_mode(`str`, *optional*):
                Output format type. "default" for resize, "crop" for shorter-length resize and centercrop.
            identity_id (`int` or `List[int]`, *optional*):
                Identity slot index (or per-sample indices) in the learnable identity token bank.
            identity_strength (`float`, *optional*, defaults to 1.0):
                Scale applied to identity tokens for conditioned branch.
            identity_negative_strength (`float`, *optional*, defaults to 0.0):
                Scale applied to identity tokens for unconditioned branch.
            update_identity_bank (`bool`, *optional*, defaults to False):
                Update the selected identity slot(s) from current conditioning latents.
            identity_update_momentum (`float`, *optional*, defaults to 0.25):
                EMA update ratio for identity bank writes.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        height, width = self.get_condition_shape(image, resolution, scale_factor_spatial=scale_factor_spatial)
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )
        assert resize_mode in ['default', 'crop'], f"Unsupported resize_mode {resize_mode}, and you can only choose from [default, crop]"

        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if emotion_guidance_scale > 0 and (emotion_id is None or emotion_intensity <= 0):
            loguru.logger.warning(
                "Emotion guidance is enabled but emotion control is missing; disabling emotion guidance for this call."
            )
            emotion_guidance_scale = 0.0


        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._emotion_guidance_scale = float(emotion_guidance_scale)
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)


        # 3. Encode inputs
        dit_dtype = self.dit.dtype
        identity_token_count = (
            self.identity_tokens_per_id
            if self.identity_bank_enabled and identity_id is not None
            else 0
        )

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds, 
                prompt_attention_mask, 
                negative_prompt_embeds, 
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._append_identity_tokens(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                identity_id=identity_id,
                identity_strength=identity_strength,
                identity_negative_strength=identity_negative_strength,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_seq_len = max_sequence_length + identity_token_count
            effective_batch_size = batch_size * num_videos_per_prompt
            prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_base_embs = self._prepare_audio_emb_for_dit(
            audio_emb,
            num_frames=num_frames,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_cond_embs, emotion_active = self._apply_emotion_channel(
            audio_emb=audio_base_embs,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_guidance_embs = None
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_base_embs)
            if emotion_active and self.emotion_guidance_scale > 0.0:
                audio_guidance_embs = audio_base_embs
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)
        
        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        image = self.video_processor.preprocess(image, height=height, width=width, resize_mode=resize_mode)
        image = image.to(device=device, dtype=prompt_embeds.dtype)

        num_channels_latents = self.dit.config.in_channels
            
        latents = self.prepare_latents(
            image=image, 
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=1, 
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        if update_identity_bank and identity_id is not None:
            try:
                # Cond image latent sits in the first conditioned temporal slot.
                self.register_identity_from_latents(
                    identity_id=identity_id,
                    latents=latents[:, :, :1],
                    momentum=identity_update_momentum,
                )
                if identity_token_count > 0:
                    prompt_embeds, negative_prompt_embeds = self._refresh_identity_tokens(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        identity_id=identity_id,
                        identity_strength=identity_strength,
                        identity_negative_strength=identity_negative_strength,
                        batch_size=batch_size,
                        num_videos_per_prompt=num_videos_per_prompt,
                    )
            except Exception as exc:
                loguru.logger.warning(
                    "Identity bank update (AI2V) failed; continuing without update. Error: {}",
                    exc,
                )

        # 6. Prepare ref_target_masks to latent size
        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)

        # 7. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    raise GenerationInterrupted()

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                timestep[:, :1] = 0

                noise_pred_cond = self._predict_avatar_noise(
                    hidden_states=latents,
                    timestep=timestep[: latents.shape[0]],
                    encoder_hidden_states=prompt_embeds[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_attention_mask,
                    num_cond_latents=1,
                    audio_embs=audio_cond_embs[latents.shape[0] :] if self.do_classifier_free_guidance else audio_cond_embs,
                    ref_target_masks=ref_target_masks,
                )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    timestep_uncond[:, :1] = 0

                    noise_pred_uncond = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=1,
                        audio_embs=audio_unond_embs,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_text = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                        encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                        num_cond_latents=1,
                        audio_embs=audio_unond_embs,
                        ref_target_masks=ref_target_masks,
                    )

                    if emotion_active and self.emotion_guidance_scale > 0.0 and audio_guidance_embs is not None:
                        noise_pred_audio = self._predict_avatar_noise(
                            hidden_states=latents,
                            timestep=timestep_uncond,
                            encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                            encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                            num_cond_latents=1,
                            audio_embs=audio_guidance_embs,
                            ref_target_masks=ref_target_masks,
                        )
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_audio - noise_pred_text)
                            + self.emotion_guidance_scale * (noise_pred_cond - noise_pred_audio)
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_cond - noise_pred_text)
                        )
                else:
                    noise_pred = noise_pred_cond

                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents[:, :, 1:] = self.scheduler.step(noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self.vae.decode(latents, return_dict=False)[0]
        output_video = self._apply_hybrid_mouth_renderer(
            decoded_video=output_video,
            mouth_zone_masks=mouth_zone_masks,
            resize_mode=resize_mode,
        )
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else: 
            return output_video


    @torch.no_grad()
    def generate_avc(
        self,
        video: List[Image.Image],
        video_latent: torch.Tensor,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 93,
        num_cond_frames: int = 13,
        num_inference_steps: int = 50,
        use_distill: bool = False,
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        use_kv_cache=True,
        offload_kv_cache=False,
        enhance_hf=True,
        # avatar related params
        audio_emb: torch.Tensor = None,
        ref_latent: torch.Tensor = None,
        ref_img_index: int = None,
        mask_frame_range: int = None,
        ref_target_masks: torch.Tensor = None,
        resize_mode: Optional[str] = "crop", # "default" / "crop"
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        update_identity_bank: bool = False,
        identity_update_momentum: float = 0.25,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
    ):
        r"""
        Generates video frames from a source video and text prompt using diffusion process with spatio-temporal conditioning.

        Args:
            video (`List[Image.Image]`):
                Input video frames for conditioning.
            prompt (`str or List[str]`, *optional*):
                Text prompt(s) for video content generation.
            negative_prompt (`str or List[str]`, *optional*):
                Negative prompt(s) for content exclusion. If not provided, uses empty string.
            num_frames (`int`, *optional*, defaults to 93):
                Number of frames to generate for the video. Should satisfy (num_frames - 1) % vae_scale_factor_temporal == 0.
            num_cond_frames (`int`, *optional*, defaults to 13):
                Number of conditioning frames from the input video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation.
            use_distill (`bool`, *optional*, defaults to False):
                Whether to use distillation sampling schedule.
            text_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            audio_guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale. Controls audio adherence. Larger values may lead to exaggerated mouth.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos to generate per prompt.
            generator (`torch.Generator or List[torch.Generator]`, *optional*):
                Random seed generator(s) for noise generation.
            latents (`torch.Tensor`, *optional*):
                Precomputed latent tensor. If not provided, random latents are generated.
            output_type (`str`, *optional*, defaults to "np"):
                Output format type. "np" for numpy array, "latent" for latent tensor.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional attention parameters for the model.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length for text encoding.
            use_kv_cache (`bool`, *optional*, defaults to True):
                Whether to use key-value cache for faster inference.
            offload_kv_cache (`bool`, *optional*, defaults to False):
                Whether to offload key-value cache to CPU to save VRAM.
            enhance_hf (`bool`, *optional*, defaults to True):
                Whether to use enhanced high-frequency denoising schedule.
            audio_emb (`torch.Tensor`):
                Audio embedding to driven the lip movements and body motions of character.
            ref_latent (`torch.Tensor`):
                The latent of reference anchor image when generate long video.
            ref_img_index (`int`, *optional*, defaults to 10)
                The insertion position of the reference image relative to the noisy latent along the temporal dimension.
            mask_frame_range (`int`, *optional*, defaults to 0)
                The attention masking range for the reference image.
            ref_target_masks(`torch.Tensor`, *optional*, defaults to None):
                Mask used in dual-speaker audio-driven mode.
            resize_mode(`str`, *optional*):
                Output format type. "default" for resize, "crop" for shorter-length resize and centercrop.
            identity_id (`int` or `List[int]`, *optional*):
                Identity slot index (or per-sample indices) in the learnable identity token bank.
            identity_strength (`float`, *optional*, defaults to 1.0):
                Scale applied to identity tokens for conditioned branch.
            identity_negative_strength (`float`, *optional*, defaults to 0.0):
                Scale applied to identity tokens for unconditioned branch.
            update_identity_bank (`bool`, *optional*, defaults to False):
                Update the selected identity slot(s) from current conditioning latents.
            identity_update_momentum (`float`, *optional*, defaults to 0.25):
                EMA update ratio for identity bank writes.

        Returns:
            np.ndarray or torch.Tensor:
                Generated video frames. If output_type is "np", returns numpy array of shape (B, N, H, W, C).
                If output_type is "latent", returns latent tensor.
        """

        # 1. Check inputs. Raise error if not correct
        assert not (use_distill and enhance_hf), "use_distill and enhance_hf cannot both be True"
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            scale_factor_spatial
        )
        assert resize_mode in ['default', 'crop'], f"Unsupported resize_mode {resize_mode}, and you can choose from [default, crop]"
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            loguru.logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if emotion_guidance_scale > 0 and (emotion_id is None or emotion_intensity <= 0):
            loguru.logger.warning(
                "Emotion guidance is enabled but emotion control is missing; disabling emotion guidance for this call."
            )
            emotion_guidance_scale = 0.0

        self._text_guidance_scale = text_guidance_scale
        self._audio_guidance_scale = audio_guidance_scale
        self._emotion_guidance_scale = float(emotion_guidance_scale)
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self.device

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # 3. Encode inputs
        dit_dtype = self.dit.dtype
        identity_token_count = (
            self.identity_tokens_per_id
            if self.identity_bank_enabled and identity_id is not None
            else 0
        )

        if context_parallel_util.get_cp_rank() == 0:
            (
                prompt_embeds, 
                prompt_attention_mask, 
                negative_prompt_embeds, 
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dit_dtype,
                device=device,
            )
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self._append_identity_tokens(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                identity_id=identity_id,
                identity_strength=identity_strength,
                identity_negative_strength=identity_negative_strength,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
            )
            if context_parallel_util.get_cp_size() > 1:
                context_parallel_util.cp_broadcast(prompt_embeds)
                context_parallel_util.cp_broadcast(prompt_attention_mask)
                if self.do_classifier_free_guidance:
                    context_parallel_util.cp_broadcast(negative_prompt_embeds)
                    context_parallel_util.cp_broadcast(negative_prompt_attention_mask)
        elif context_parallel_util.get_cp_size() > 1:
            caption_channels = self.text_encoder.config.d_model
            prompt_seq_len = max_sequence_length + identity_token_count
            effective_batch_size = batch_size * num_videos_per_prompt
            prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
            prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
            context_parallel_util.cp_broadcast(prompt_embeds)
            context_parallel_util.cp_broadcast(prompt_attention_mask)
            if self.do_classifier_free_guidance:
                negative_prompt_embeds = torch.zeros([effective_batch_size, 1, prompt_seq_len, caption_channels], dtype=dit_dtype, device=device)
                negative_prompt_attention_mask = torch.zeros([effective_batch_size, prompt_seq_len], dtype=torch.int64, device=device)
                context_parallel_util.cp_broadcast(negative_prompt_embeds)
                context_parallel_util.cp_broadcast(negative_prompt_attention_mask)

        audio_base_embs = self._prepare_audio_emb_for_dit(
            audio_emb,
            num_frames=num_frames,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_cond_embs, emotion_active = self._apply_emotion_channel(
            audio_emb=audio_base_embs,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            device=device,
        )
        audio_cache_embs = audio_base_embs
        audio_guidance_embs = None
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            audio_unond_embs = torch.zeros_like(audio_base_embs)
            if emotion_active and self.emotion_guidance_scale > 0.0:
                audio_guidance_embs = audio_base_embs
            audio_cond_embs = torch.cat([audio_cond_embs, audio_cond_embs], dim=0)

        # 4. Prepare timesteps
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps

        if enhance_hf:
            tail_uniform_start = 500
            tail_uniform_end = 0
            num_tail_uniform_steps = 10
            timesteps_uniform_tail = list(np.linspace(tail_uniform_start, tail_uniform_end, num_tail_uniform_steps, dtype=np.float32, endpoint=(tail_uniform_end != 0)))
            timesteps_uniform_tail = [torch.tensor(t, device=device).unsqueeze(0) for t in timesteps_uniform_tail]
            filtered_timesteps = [timestep.unsqueeze(0) for timestep in timesteps if timestep > tail_uniform_start]
            timesteps = torch.cat(filtered_timesteps + timesteps_uniform_tail)
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=timesteps.device)])

        # 5. Prepare latent variables
        video = self.video_processor.preprocess_video(video, height=height, width=width, resize_mode=resize_mode)
        video = video.to(device=device, dtype=prompt_embeds.dtype) 
        cond_videos = video[:, :, -num_cond_frames:]
        cond_videos_latents = retrieve_latents(self.vae.encode(cond_videos), generator, sample_mode="argmax")
        cond_videos_latents = self.normalize_latents(cond_videos_latents)
        if update_identity_bank and identity_id is not None:
            try:
                self.register_identity_from_latents(
                    identity_id=identity_id,
                    latents=cond_videos_latents,
                    momentum=identity_update_momentum,
                )
                if identity_token_count > 0:
                    prompt_embeds, negative_prompt_embeds = self._refresh_identity_tokens(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        identity_id=identity_id,
                        identity_strength=identity_strength,
                        identity_negative_strength=identity_negative_strength,
                        batch_size=batch_size,
                        num_videos_per_prompt=num_videos_per_prompt,
                    )
            except Exception as exc:
                loguru.logger.warning(
                    "Identity bank update (AVC) failed; continuing without update. Error: {}",
                    exc,
                )


        num_channels_latents = self.dit.config.in_channels
        latents = self.prepare_latents(
            video=video_latent,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            dtype=dit_dtype,
            device=device,
            generator=generator,
            latents=latents,
            need_encode=False
        )
        if context_parallel_util.get_cp_size() > 1:
            context_parallel_util.cp_broadcast(latents)

        output_num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
        cache_num_cond_latents = output_num_cond_latents
        
        # 6. Prepare ref_target_masks from source size to latent size
        if ref_target_masks is not None:
            ref_target_masks = self._resize_and_centercrop_tensor(ref_target_masks, height, width, resize_mode)

        # 7. Add reference image
        num_ref_latents = 0
        if ref_latent is not None:
            cache_num_cond_latents += 1
            num_ref_latents = 1
            latents = torch.cat([ref_latent, latents], dim=2)

        # 8. Denoising loop
        if context_parallel_util.get_cp_size() > 1:
            torch.distributed.barrier(group=context_parallel_util.get_cp_group())

        if use_kv_cache:
            cond_latents = latents[:, :, :cache_num_cond_latents]
            kv_cache_num_cond_latents = self._cache_clean_latents(cond_latents, max_sequence_length, offload_kv_cache=offload_kv_cache, device=self.device, dtype=dit_dtype, \
                audio_embs=audio_cache_embs, num_cond_latents=cache_num_cond_latents, num_ref_latents=num_ref_latents, ref_img_index=ref_img_index)
            kv_cache_dict = self._get_kv_cache_dict()
            latents = latents[:, :, cache_num_cond_latents:]
            active_num_cond_latents = kv_cache_num_cond_latents
        else:
            kv_cache_dict = {}
            active_num_cond_latents = cache_num_cond_latents

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    raise GenerationInterrupted()

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(dit_dtype)

                timestep = t.expand(latent_model_input.shape[0]).to(dit_dtype)
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if not use_kv_cache:
                    timestep[:, :active_num_cond_latents] = 0
                
                noise_pred_cond = self._predict_avatar_noise(
                    hidden_states=latents,
                    timestep=timestep[: latents.shape[0]],
                    encoder_hidden_states=prompt_embeds[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask[latents.shape[0] :] if self.do_classifier_free_guidance else prompt_attention_mask,
                    num_cond_latents=active_num_cond_latents,
                    kv_cache_dict=kv_cache_dict,
                    audio_embs=audio_cond_embs[latents.shape[0] :] if self.do_classifier_free_guidance else audio_cond_embs,
                    num_ref_latents=num_ref_latents,
                    ref_img_index=ref_img_index,
                    mask_frame_range=mask_frame_range,
                    ref_target_masks=ref_target_masks,
                )

                if self.do_classifier_free_guidance:
                    timestep_uncond = t.expand(latents.shape[0]).to(dit_dtype)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    if not use_kv_cache:
                        timestep_uncond[:, :active_num_cond_latents] = 0

                    noise_pred_uncond = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=negative_prompt_embeds,
                        encoder_attention_mask=negative_prompt_attention_mask,
                        num_cond_latents=active_num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_unond_embs,
                        num_ref_latents=num_ref_latents, 
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )
                    noise_pred_text = self._predict_avatar_noise(
                        hidden_states=latents,
                        timestep=timestep_uncond,
                        encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                        encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                        num_cond_latents=active_num_cond_latents,
                        kv_cache_dict=kv_cache_dict,
                        audio_embs=audio_unond_embs,
                        num_ref_latents=num_ref_latents,
                        ref_img_index=ref_img_index,
                        mask_frame_range=mask_frame_range,
                        ref_target_masks=ref_target_masks,
                    )

                    if emotion_active and self.emotion_guidance_scale > 0.0 and audio_guidance_embs is not None:
                        noise_pred_audio = self._predict_avatar_noise(
                            hidden_states=latents,
                            timestep=timestep_uncond,
                            encoder_hidden_states=prompt_embeds[latents.shape[0] :],
                            encoder_attention_mask=prompt_attention_mask[latents.shape[0] :],
                            num_cond_latents=active_num_cond_latents,
                            kv_cache_dict=kv_cache_dict,
                            audio_embs=audio_guidance_embs,
                            num_ref_latents=num_ref_latents,
                            ref_img_index=ref_img_index,
                            mask_frame_range=mask_frame_range,
                            ref_target_masks=ref_target_masks,
                        )
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_audio - noise_pred_text)
                            + self.emotion_guidance_scale * (noise_pred_cond - noise_pred_audio)
                        )
                    else:
                        noise_pred = (
                            noise_pred_uncond
                            + text_guidance_scale * (noise_pred_text - noise_pred_uncond)
                            + audio_guidance_scale * (noise_pred_cond - noise_pred_text)
                        )
                else:
                    noise_pred = noise_pred_cond
                
                # negate for scheduler compatibility
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                if use_kv_cache:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    latents[:, :, active_num_cond_latents:] = self.scheduler.step(noise_pred[:, :, active_num_cond_latents:], t, latents[:, :, active_num_cond_latents:], return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
            
            if use_kv_cache:
                latents = torch.cat([cond_latents, latents], dim=2)
            
            if ref_latent is not None:
                latents = latents[:, :, num_ref_latents:]

            latents[:, :, :output_num_cond_latents] = cond_videos_latents

        self._current_timestep = None

        if output_type == 'latent':
            return latents
        
        if output_type == 'both':
            latents_ = latents.clone()

        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        output_video = self.vae.decode(latents, return_dict=False)[0]
        output_video = self._apply_hybrid_mouth_renderer(
            decoded_video=output_video,
            mouth_zone_masks=mouth_zone_masks,
            resize_mode=resize_mode,
        )
        output_video = self.video_processor.postprocess_video(output_video)

        if output_type == 'both':
            return (output_video, latents_)
        else: 
            return output_video
    
    @torch.no_grad()
    def generate_streaming_ai2v(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        audio_stream=None,  # Generator yielding audio chunks
        resolution: Literal["480p", "720p"] = "480p",
        num_frames: int = 93,
        num_inference_steps: int = 8,  # Distilled: 8 steps instead of 50
        text_guidance_scale: float = 4.0,
        audio_guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        audio_emb: torch.Tensor = None,
        resize_mode: Optional[str] = "crop",
        identity_id: Optional[Union[int, List[int], torch.Tensor]] = None,
        identity_strength: float = 1.0,
        identity_negative_strength: float = 0.0,
        emotion_id: Optional[Union[int, str, List[Union[int, str]], torch.Tensor]] = None,
        emotion_intensity: float = 0.0,
        emotion_guidance_scale: float = 0.0,
        mouth_zone_masks: Optional[torch.Tensor] = None,
    ):
        r"""
        Streaming-like video generation (Image-to-Video).
        Yields video frames progressively after latent denoising is complete.
        
        Args:
            image: Input image for video generation.
            prompt: Text prompt(s) for video content generation.
            audio_stream: Optional generator yielding audio chunks [sample_rate=16000].
            resolution: "480p" or "720p".
            num_frames: Number of frames to generate.
            num_inference_steps: Denoising steps (8 = distilled fast mode).
            text_guidance_scale: CFG scale for text.
            audio_guidance_scale: CFG scale for audio.
            generator: Random seed generator.
            audio_emb: Pre-computed audio embedding (alternative to audio_stream).
            resize_mode: "default" or "crop".
            identity_id: Identity slot index (or per-sample indices) in identity token bank.
            identity_strength: Scale applied to identity tokens for conditioned branch.
            identity_negative_strength: Scale applied to identity tokens for unconditioned branch.
            emotion_id: Emotion class id or label.
            emotion_intensity: Emotion intensity multiplier.
            emotion_guidance_scale: Separate CFG scale for emotion channel.
        
        Yields:
            np.ndarray: Frame as numpy array [H, W, 3] in range [0, 255].
        """
        
        scale_factor_spatial = self.vae_scale_factor_spatial * 2
        if self.dit.cp_split_hw is not None:
            scale_factor_spatial *= max(self.dit.cp_split_hw)
        
        height, width = self.get_condition_shape(image, resolution, scale_factor_spatial=scale_factor_spatial)
        self.check_inputs(prompt, None, height, width, scale_factor_spatial)
        
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
        
        device = self.device
        
        # 1. Resolve audio embedding.
        if audio_emb is None:
            if audio_stream is None:
                raise ValueError("Either `audio_stream` or `audio_emb` must be provided.")

            audio_chunks = []
            sample_rate = 16000
            for chunk in audio_stream:
                if chunk is None:
                    continue
                audio_chunks.append(np.asarray(chunk, dtype=np.float32))

            if not audio_chunks:
                raise ValueError("`audio_stream` yielded no chunks.")

            full_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32, copy=False)
            audio_stride = max(int(self.vae_scale_factor_temporal), 1)
            full_audio_emb = self.get_audio_embedding(
                full_audio,
                fps=16 * audio_stride,
                device=device,
                sample_rate=sample_rate,
            )
            audio_emb = self._build_windowed_audio_embedding(
                full_audio_emb,
                num_frames=num_frames,
                device=device,
            )
        else:
            audio_emb = self._prepare_audio_emb_for_dit(
                audio_emb,
                num_frames=num_frames,
                batch_size=1,
                num_videos_per_prompt=1,
                device=device,
            )

        # 2. Run full generate_ai2v with output_type='latent' (image + audio conditioning)
        latents = self.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt="",
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            use_distill=(num_inference_steps <= 16),
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type="latent",
            max_sequence_length=max_sequence_length,
            audio_emb=audio_emb,
            resize_mode=resize_mode,
            identity_id=identity_id,
            identity_strength=identity_strength,
            identity_negative_strength=identity_negative_strength,
            emotion_id=emotion_id,
            emotion_intensity=emotion_intensity,
            emotion_guidance_scale=emotion_guidance_scale,
            mouth_zone_masks=mouth_zone_masks,
        )
        
        # 3. Stream decode: denormalize, decode frame-by-frame, yield
        latents = latents.to(self.vae.dtype)
        latents = self.denormalize_latents(latents)
        vae_decoder = StreamingVAEDecoder(self.vae, chunk_size=1, enable_amp=True)
        frame_times = []
        stream_mouth_mask = None
        stream_boundary_mask = None
        prev_stabilized = None
        prev_stabilized_for_flicker = None
        hybrid_artifacts = []
        hybrid_boundary_diffs = []
        hybrid_global_diffs = []
        for frame_idx, decoded in enumerate(vae_decoder.decode_streaming(latents)):
            frame_time = time.time()
            frame_tensor = decoded
            if mouth_zone_masks is not None and self.hybrid_renderer_enabled:
                if stream_mouth_mask is None:
                    prepared = self._prepare_mouth_zone_mask(
                        mouth_zone_masks=mouth_zone_masks,
                        batch_size=decoded.shape[0],
                        num_frames=1,
                        height=decoded.shape[-2],
                        width=decoded.shape[-1],
                        device=decoded.device,
                        dtype=decoded.dtype,
                        resize_mode=resize_mode,
                    )
                    if prepared is not None:
                        stream_mouth_mask = prepared[:, :, 0]
                        stream_boundary_mask = self._compute_seam_boundary_mask(prepared)[:, :, 0]

                if stream_mouth_mask is not None and stream_boundary_mask is not None:
                    branch = self._build_mouth_controlled_branch(
                        decoded.unsqueeze(2),
                        strength=float(self.hybrid_renderer_mouth_strength),
                    )[:, :, 0]
                    blended = decoded * (1.0 - stream_mouth_mask) + branch * stream_mouth_mask
                    if prev_stabilized is not None:
                        a = float(self.hybrid_renderer_temporal_alpha)
                        blended = (
                            blended * (1.0 - stream_boundary_mask)
                            + (a * blended + (1.0 - a) * prev_stabilized) * stream_boundary_mask
                        )
                    prev_stabilized = blended.detach()
                    frame_tensor = blended

                    hybrid_artifacts.append(float((torch.abs(frame_tensor - decoded) * stream_mouth_mask).mean().item()))
                    if prev_stabilized_for_flicker is not None:
                        diff = torch.abs(frame_tensor - prev_stabilized_for_flicker)
                        hybrid_boundary_diffs.append(float((diff * stream_boundary_mask).mean().item()))
                        hybrid_global_diffs.append(float(diff.mean().item()))
                    prev_stabilized_for_flicker = frame_tensor.detach()

            frame_np = (frame_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frame_times.append(time.time() - frame_time)
            yield frame_np
        
        # 4. Log performance
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            sorted_times = sorted(frame_times)
            p95_latency = sorted_times[int(len(sorted_times) * 0.95)] * 1000 if sorted_times else 0
            self.metrics.record('streaming_fps', fps)
            self.metrics.record('streaming_p95_latency_ms', p95_latency)
            if hybrid_artifacts:
                artifact_mean = float(sum(hybrid_artifacts) / len(hybrid_artifacts))
                self.metrics.record("hybrid_stream_artifact_energy", artifact_mean)
            if hybrid_boundary_diffs and hybrid_global_diffs:
                boundary_mean = float(sum(hybrid_boundary_diffs) / len(hybrid_boundary_diffs))
                global_mean = float(sum(hybrid_global_diffs) / len(hybrid_global_diffs))
                ratio = boundary_mean / max(global_mean, 1e-6)
                self.metrics.record("hybrid_stream_flicker_ratio", ratio)
                self.metrics.record("hybrid_stream_budget_ok", int(
                    ratio <= float(self.hybrid_renderer_flicker_budget)
                ))
            if context_parallel_util.get_cp_rank() == 0:
                loguru.logger.info(f"Streaming complete: {fps:.1f} FPS, P95 latency: {p95_latency:.1f}ms")

    

    def to(self, device: str | torch.device):
        """
        Move pipeline to specified device.

        Args:
            device: Target device string

        Returns:
            Self
        """
        self.device = device
        if self.dit is not None:
            self.dit = self.dit.to(device, non_blocking=True)
            if hasattr(self.dit, 'lora_dict') and self.dit.lora_dict:
                for lora_key, lora_network in self.dit.lora_dict.items():
                    for lora in lora_network.loras:
                        lora.to(device, non_blocking=True)
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device, non_blocking=True)
        if self.vae is not None:
            self.vae = self.vae.to(device, non_blocking=True)
        if self.identity_embedding is not None:
            self.identity_embedding = self.identity_embedding.to(device, non_blocking=True)
        if self.identity_latent_projector is not None:
            self.identity_latent_projector = self.identity_latent_projector.to(device, non_blocking=True)
        if self.phoneme_proj is not None:
            self.phoneme_proj = self.phoneme_proj.to(device, non_blocking=True)
        if self.phoneme_alignment_head is not None:
            self.phoneme_alignment_head = self.phoneme_alignment_head.to(device, non_blocking=True)
        if self.emotion_embedding is not None:
            self.emotion_embedding = self.emotion_embedding.to(device, non_blocking=True)
        if self.emotion_proj is not None:
            self.emotion_proj = self.emotion_proj.to(device, non_blocking=True)
        return self
    
ArachneXVideoAvatarPipeline = LongCatVideoAvatarPipeline


