"""
WebDataset iterable for ARACHNE-X ``train.py`` / LoRA training.

Shard layout (written by ``scripts/pack_latent_shards_wds.py``):

- ``__key__``: unique id
- ``sample.pt``: bytes of ``torch.save`` dict (same keys as flat .pt training files)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch.utils.data import IterableDataset

try:
    import webdataset as wds
except ImportError as e:
    wds = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _require_webdataset() -> None:
    if wds is None:
        raise ImportError(
            "WebDataset training requires: pip install webdataset\n"
            f"Original error: {_IMPORT_ERR}"
        ) from _IMPORT_ERR


class LatentWebDataset(IterableDataset):
    """
    Infinite resampled stream over shards; compatible with ``DataLoader`` + custom collate.

    ``url`` example: ``/data/shards/shard-{000000..000099}.tar`` or ``/data/shards/*.tar``
    (brace ranges preferred for exact shard lists).
    """

    def __init__(
        self,
        url: str,
        *,
        require_audio: bool,
        shuffle: int = 5000,
        decode: Optional[Callable[[Dict], Dict[str, torch.Tensor]]] = None,
    ):
        _require_webdataset()
        super().__init__()
        self.url = url
        self.require_audio = require_audio
        self.shuffle = shuffle
        self._decode = decode

    def __iter__(self):
        _require_webdataset()
        from arachne_x.training_latent_common import decode_wds_sample_pt

        decoder = self._decode
        if decoder is None:
            require_audio = self.require_audio

            def decoder(sample):
                return decode_wds_sample_pt(sample, require_audio=require_audio)

        wd_kwargs = dict(resampled=True, handler=wds.warn_and_continue)
        node_split = getattr(wds, "split_by_node", None)
        worker_split = getattr(wds, "split_by_worker", None)
        if node_split is not None:
            wd_kwargs["nodesplit"] = node_split
        if worker_split is not None:
            wd_kwargs["workersplit"] = worker_split
        dataset = wds.WebDataset(self.url, **wd_kwargs)
        if self.shuffle > 0:
            dataset = dataset.shuffle(self.shuffle)
        dataset = dataset.map(decoder)
        yield from dataset
