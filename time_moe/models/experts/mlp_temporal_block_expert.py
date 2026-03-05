from transformers.activations import ACT2FN
from torch import nn
import torch

from .base import BaseTokenExpert
from .common import ExpertRMSNorm


class MLPTemporalBlockExpert(BaseTokenExpert):
    expert_type = "generic"
    interface_kind = "flat"

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            output_norm: bool = True,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.output_norm = ExpertRMSNorm(hidden_size) if output_norm else nn.Identity()

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.output_norm(y)
