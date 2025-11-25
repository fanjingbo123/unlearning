import logging
from typing import Iterable, List

import torch.nn as nn


_DEFAULT_KEYWORDS: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "c_attn",
    "W_pack",
    "out_proj",
]


def detect_lora_target_modules(model: nn.Module, extra_keywords: Iterable[str] = ()) -> List[str]:
    """Heuristically discover attention projection modules suitable for LoRA.

    Args:
        model: Loaded CausalLM model whose submodules will be inspected.
        extra_keywords: Optional iterable of additional substring patterns to match.

    Returns:
        Sorted list of keyword substrings that exist in the model's Linear submodules.

    Raises:
        ValueError: If no candidate modules are found.
    """

    keywords = list(_DEFAULT_KEYWORDS) + list(extra_keywords)
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for kw in keywords:
                if kw in name:
                    found.add(kw)
    if not found:
        raise ValueError(
            "No target LoRA modules found. Please pass extra_keywords to match this model's attention layers."
        )
    logging.info("Detected LoRA target modules: %s", sorted(found))
    return sorted(found)
