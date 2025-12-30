from dataclasses import dataclass
import torch


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    sleeve: torch.Tensor
    pocket: torch.Tensor
    buttons: torch.Tensor
