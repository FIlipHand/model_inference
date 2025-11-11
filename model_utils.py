from dataclasses import dataclass
from typing import List

import torch


@dataclass
class KVCache:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class ModelOutput:
    logits: torch.FloatTensor
    past_key_values: List[KVCache]
