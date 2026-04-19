import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict, Optional


class PhonemeTemporalAligner:
    """
    Lightweight phoneme-like temporal aligner.
    Produces frame-level pseudo-phoneme probabilities aligned to target length.
    """

    def __init__(
        self,
        num_phonemes: int = 10,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
    ):
        self.num_phonemes = num_phonemes
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms

    @staticmethod
    def _normalize_feature(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo = np.percentile(x, 5)
        hi = np.percentile(x, 95)
        return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    @staticmethod
    def _frame_signal(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if len(audio) < frame_len:
            pad = frame_len - len(audio)
            audio = np.pad(audio, (0, pad))

        n_frames = 1 + (len(audio) - frame_len) // hop_len
        if n_frames <= 0:
            n_frames = 1

        shape = (n_frames, frame_len)
        strides = (audio.strides[0] * hop_len, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        return np.ascontiguousarray(frames)

    def _extract_frame_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        frame_len = max(int(sample_rate * (self.frame_ms / 1000.0)), 16)
        hop_len = max(int(sample_rate * (self.hop_ms / 1000.0)), 8)

        frames = self._frame_signal(audio, frame_len=frame_len, hop_len=hop_len)
        window = np.hanning(frame_len).astype(np.float32)
        frames_w = frames * window[None, :]

        energy = np.sqrt(np.mean(frames_w**2, axis=1) + 1e-8)

        signs = np.sign(frames)
        zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1).astype(np.float32)

        spectrum = np.abs(np.fft.rfft(frames_w, axis=1)).astype(np.float32) + 1e-8
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate).astype(np.float32)
        spec_sum = spectrum.sum(axis=1) + 1e-8

        centroid = (spectrum * freqs[None, :]).sum(axis=1) / spec_sum
        bandwidth = np.sqrt(
            ((freqs[None, :] - centroid[:, None]) ** 2 * spectrum).sum(axis=1) / spec_sum
        )

        cumsum = np.cumsum(spectrum, axis=1)
        roll_idx = (cumsum >= (0.85 * spec_sum[:, None])).argmax(axis=1)
        rolloff = freqs[roll_idx]

        spec_norm = spectrum / (np.linalg.norm(spectrum, axis=1, keepdims=True) + 1e-8)
        flux = np.zeros(spec_norm.shape[0], dtype=np.float32)
        if spec_norm.shape[0] > 1:
            flux[1:] = np.sqrt(((spec_norm[1:] - spec_norm[:-1]) ** 2).sum(axis=1))

        return {
            "energy": self._normalize_feature(energy),
            "zcr": self._normalize_feature(zcr),
            "centroid": self._normalize_feature(centroid),
            "bandwidth": self._normalize_feature(bandwidth),
            "rolloff": self._normalize_feature(rolloff),
            "flux": self._normalize_feature(flux),
        }

    def _classify_frames(self, feat: Dict[str, np.ndarray]) -> np.ndarray:
        energy = feat["energy"]
        zcr = feat["zcr"]
        centroid = feat["centroid"]
        bandwidth = feat["bandwidth"]
        rolloff = feat["rolloff"]
        flux = feat["flux"]

        n = len(energy)
        cls = np.full(n, 9, dtype=np.int64)  # default: unknown

        silence = energy < 0.08
        voiced = (energy > 0.12) & (zcr < 0.35) & (centroid < 0.7)
        unvoiced = ~silence & ~voiced

        cls[silence] = 0  # silence

        # voiced classes
        vowel_open = voiced & (centroid < 0.35) & (bandwidth < 0.45)
        vowel_mid = voiced & (centroid >= 0.35) & (centroid < 0.55)
        vowel_close = voiced & (centroid >= 0.55)
        nasal = voiced & (flux > 0.55) & (zcr < 0.2)
        liquid_glide = voiced & (bandwidth > 0.65) & (rolloff > 0.7)

        cls[vowel_open] = 1
        cls[vowel_mid] = 2
        cls[vowel_close] = 3
        cls[nasal] = 6
        cls[liquid_glide] = 7

        # unvoiced classes
        fricative = unvoiced & ((zcr > 0.6) | (centroid > 0.75))
        plosive = unvoiced & (flux > 0.55) & ~fricative
        transition = unvoiced & ~fricative & ~plosive

        cls[fricative] = 4
        cls[plosive] = 5
        cls[transition] = 8

        return cls

    def _soften_probs(self, probs: np.ndarray) -> np.ndarray:
        if probs.shape[0] < 3:
            return probs
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        out = np.zeros_like(probs, dtype=np.float32)
        padded = np.pad(probs, ((1, 1), (0, 0)), mode="edge")
        for i in range(probs.shape[0]):
            out[i] = (
                kernel[0] * padded[i]
                + kernel[1] * padded[i + 1]
                + kernel[2] * padded[i + 2]
            )
        out = np.clip(out, 1e-8, None)
        out /= out.sum(axis=1, keepdims=True)
        return out

    def extract(
        self,
        speech_array: np.ndarray,
        sample_rate: int = 16000,
        target_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        audio = np.asarray(speech_array, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise ValueError("Empty audio array in PhonemeTemporalAligner.extract")

        audio = np.clip(audio, -1.0, 1.0)
        feat = self._extract_frame_features(audio, sample_rate=sample_rate)
        cls = self._classify_frames(feat)

        probs = np.zeros((cls.shape[0], self.num_phonemes), dtype=np.float32)
        probs[np.arange(cls.shape[0]), cls] = 1.0
        probs = self._soften_probs(probs)

        probs_t = torch.from_numpy(probs).float()
        if target_len is not None and target_len > 0 and probs_t.shape[0] != target_len:
            probs_t = probs_t.transpose(0, 1).unsqueeze(0)
            probs_t = F.interpolate(probs_t, size=target_len, mode="linear", align_corners=False)
            probs_t = probs_t.squeeze(0).transpose(0, 1).contiguous()
            probs_t = torch.clamp(probs_t, min=1e-8)
            probs_t = probs_t / probs_t.sum(dim=-1, keepdim=True)

        ids_t = probs_t.argmax(dim=-1)
        conf_t = probs_t.max(dim=-1).values

        out = {
            "phoneme_probs": probs_t,  # [T, P]
            "phoneme_ids": ids_t,      # [T]
            "confidence": conf_t,      # [T]
            "voiced_ratio": float(((ids_t >= 1) & (ids_t <= 3)).float().mean().item()),
            "silence_ratio": float((ids_t == 0).float().mean().item()),
            "fricative_ratio": float((ids_t == 4).float().mean().item()),
            "plosive_ratio": float((ids_t == 5).float().mean().item()),
        }
        return out

