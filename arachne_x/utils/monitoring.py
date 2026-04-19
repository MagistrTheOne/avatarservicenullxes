import logging
import time
import hashlib
import os
import numpy as np
from typing import List, Optional, Callable

try:
    import librosa
    from scipy.spatial.distance import cdist
except Exception:
    librosa = None

try:
    import torch
    from torchvision import models, transforms
except Exception:
    torch = None

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self):
        self.times: List[float] = []

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.times.append(time.time() - self.t0)

    def last(self) -> float:
        return self.times[-1] if self.times else 0.0

    def p95(self) -> float:
        if not self.times:
            return 0.0
        return float(np.percentile(np.array(self.times), 95))


class MetricsLogger:
    """Collects and computes metrics for monitoring."""
    def __init__(self):
        self.timers = {}
        self.records = {}

    def timeit(self, name: str):
        t = Timer()
        self.timers[name] = t
        return t

    def record(self, key: str, value):
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(value)

    def p95(self, key: str) -> Optional[float]:
        vals = self.records.get(key, [])
        if not vals:
            return None
        return float(np.percentile(np.array(vals), 95))

    def summary(self) -> dict:
        summary = {}
        for k, v in self.records.items():
            arr = np.array(v)
            summary[k] = {
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'p95': float(np.percentile(arr, 95)),
                'count': int(arr.size)
            }
        for name, t in self.timers.items():
            summary[f'timer_{name}_p95'] = t.p95()
        return summary


def compute_dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute DTW distance between two sequences of feature vectors using librosa if available."""
    if librosa is None:
        logger.warning("compute_dtw_distance requires librosa; returning inf.")
        return float("inf")
    # x: (T1, D), y: (T2, D)
    D = cdist(x, y, metric='euclidean')
    _, wp = librosa.sequence.dtw(C=D)
    # path cost
    cost = D[tuple(zip(*wp))].sum()
    return float(cost)


def compute_lpips_vgg(imgs1: List[np.ndarray], imgs2: List[np.ndarray]) -> float:
    """Compute a simple perceptual distance using pretrained VGG features (proxy for LPIPS).
    imgs are numpy uint8 HWC arrays scaled 0..255
    """
    if torch is None:
        raise RuntimeError('torch and torchvision required for LPIPS proxy')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _make_vgg16_no_pretrained():
        try:
            return models.vgg16(weights=None)
        except TypeError:
            return models.vgg16(pretrained=False)

    local_vgg_path = os.environ.get("ARACHNE_VGG16_WEIGHTS", "").strip()
    vgg_full = _make_vgg16_no_pretrained()
    if local_vgg_path:
        state_dict = torch.load(local_vgg_path, map_location=device)
        vgg_full.load_state_dict(state_dict, strict=True)
    vgg = vgg_full.features[:16].to(device).eval()
    preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    feats = []
    with torch.no_grad():
        for img in imgs1:
            t = preproc(img).unsqueeze(0).to(device)
            f = vgg(t).cpu().numpy().flatten()
            feats.append(f)
        feats2 = []
        for img in imgs2:
            t = preproc(img).unsqueeze(0).to(device)
            f = vgg(t).cpu().numpy().flatten()
            feats2.append(f)
    feats = np.stack(feats)
    feats2 = np.stack(feats2)
    return float(np.mean(np.linalg.norm(feats - feats2, axis=1)))


def sha256_of_audio_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()
