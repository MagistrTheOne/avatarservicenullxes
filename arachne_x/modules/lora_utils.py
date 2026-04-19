# References:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/bmaltais/kohya_ss

import math
import functools
from collections import defaultdict

from typing import Callable, Dict, List, Optional, Sequence, Set, Union

import torch
import torch.nn as nn


class LoRAUPParallel(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x):
        assert x.shape[-1] % len(self.blocks) == 0
        xs = torch.chunk(x, len(self.blocks), dim=-1)
        out = torch.cat([self.blocks[i](xs[i]) for i in range(len(self.blocks))], dim=-1)
        return out


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        n_seperate=1
    ):
        super().__init__()
        self.lora_name = lora_name
        
        assert org_module.__class__.__name__ == "Linear"
        in_dim = org_module.in_features
        out_dim = org_module.out_features

        if n_seperate > 1:
            assert out_dim % n_seperate == 0

        self.lora_dim = lora_dim
        if n_seperate > 1:
            self.lora_down = torch.nn.Linear(in_dim, n_seperate * self.lora_dim, bias=False)
            self.lora_up = LoRAUPParallel([torch.nn.Linear(self.lora_dim, out_dim // n_seperate, bias=False) for _ in range(n_seperate)])
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        alpha_scale = alpha / self.lora_dim
        self.register_buffer("alpha_scale", torch.tensor(alpha_scale))

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if n_seperate > 1:
            for block in self.lora_up.blocks:
                torch.nn.init.zeros_(block.weight)
        else:
            torch.nn.init.zeros_(self.lora_up.weight)
            
        self.multiplier = multiplier
        self.use_lora = True
    
    def set_use_lora(self, use_lora):
        self.use_lora = use_lora


class LoRANetwork(torch.nn.Module):
    
    LORA_PREFIX = "lora"
    LORA_HYPHEN = "___lorahyphen___"
    
    def __init__(
        self,
        model,
        lora_network_state_dict_loaded,
        multiplier: float = 1.0,
        lora_dim: int = 128,
        alpha: float = 64,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.use_lora = True
        self.lora_dim = lora_dim
        self.alpha = alpha

        print(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")

        lora_module_names = set()
        for key in lora_network_state_dict_loaded.keys():
            if key.endswith("lora_down.weight"):
                lora_name = key.split(".lora_down.weight")[0]
                lora_module_names.add(lora_name)

        loras = []
        for lora_name in lora_module_names:
            # 还原为模型中的真实模块名
            module_name = lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            # 查找模块
            try:
                module = model
                for part in module_name.split('.'):
                    module = getattr(module, part)
            except Exception as e:
                print(f"Cannot find module: {module_name}, error: {e}")
                continue
            if module.__class__.__name__ != "Linear":
                continue

            # 推断 n_seperate
            n_seperate = 1
            prefix = lora_name + ".lora_up.blocks"
            n_blocks = sum(1 for k in lora_network_state_dict_loaded if k.startswith(prefix))
            if n_blocks > 0:
                n_seperate = n_blocks

            dim = self.lora_dim
            alpha = self.alpha

            lora = LoRAModule(
                lora_name,
                module,
                self.multiplier,
                dim,
                alpha,
                n_seperate=n_seperate
            )
            loras.append(lora)
            
        self.loras = loras
        for lora in self.loras:
            self.add_module(lora.lora_name, lora)
        print(f"create LoRA for model: {len(self.loras)} modules.")

        # assertion
        names = set()
        for lora in self.loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def disapply_to(self):
        for lora in self.loras:
            lora.disapply_to()

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.loras:
            lora.multiplier = self.multiplier

    def set_use_lora(self, use_lora):
        self.use_lora = use_lora
        for lora in self.loras:
            lora.set_use_lora(use_lora)

    def prepare_optimizer_params(self, lr):
        self.requires_grad_(True)
        all_params = []

        params = []
        for lora in self.loras:
            params.extend(lora.parameters())

        param_data = {"params": params}
        param_data["lr"] = lr
        all_params.append(param_data)

        return all_params


def create_lora_network(
    transformer,
    lora_network_state_dict_loaded,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
):
    network = LoRANetwork(
        transformer,
        lora_network_state_dict_loaded,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
    )
    return network


def module_path_to_lora_prefix(module_name: str) -> str:
    """Mirror of LoRANetwork reverse mapping: dotted path -> lora___lorahyphen___... key prefix."""
    return "lora___lorahyphen___" + module_name.replace(".", "___lorahyphen___")


def build_initial_lora_state_dict(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    name_filter: Callable[[str, nn.Linear], bool],
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Build a state dict that LoRANetwork.load_state_dict(strict=True) can consume after
    create_lora_network(...): for each matched nn.Linear, lora_down / lora_up / alpha_scale.
    """
    state: Dict[str, torch.Tensor] = {}
    scale = float(alpha) / float(rank) if rank else 1.0
    for name, mod in model.named_modules():
        if type(mod) is not nn.Linear:
            continue
        if not name_filter(name, mod):
            continue
        prefix = module_path_to_lora_prefix(name)
        in_f = mod.in_features
        out_f = mod.out_features
        down = torch.empty(rank, in_f, dtype=dtype)
        torch.nn.init.kaiming_uniform_(down, a=math.sqrt(5))
        up = torch.zeros(out_f, rank, dtype=dtype)
        state[f"{prefix}.lora_down.weight"] = down.contiguous()
        state[f"{prefix}.lora_up.weight"] = up.contiguous()
        state[f"{prefix}.alpha_scale"] = torch.tensor(scale, dtype=dtype)
    if not state:
        raise ValueError("build_initial_lora_state_dict: no Linear layers matched name_filter")
    return state


def default_avatar_train_lora_filter(
    name: str,
    mod: nn.Linear,
    include_prefixes: Optional[Union[Sequence[str], Set[str]]] = None,
) -> bool:
    """
    Default scope for avatar DiT LoRA training: transformer blocks, audio projection head,
    and final layer. Skips x_embedder (Conv3d patch stem) and text/timestep embedders unless
    include_prefixes overrides.
    """
    del mod
    if name.startswith("x_embedder"):
        return False
    if include_prefixes is not None:
        return any(name.startswith(p) for p in include_prefixes)
    return (
        name.startswith("blocks.")
        or name.startswith("audio_proj.")
        or name.startswith("final_layer.")
    )

