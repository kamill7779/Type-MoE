from torch import nn
import torch

from .base import BaseTokenExpert
from .common import ExpertRMSNorm
from .registry import register_expert


@register_expert("nbeats_trend")
class NBeatsTokenExpert(BaseTokenExpert):
    expert_type = "trend"
    interface_kind = "flat"

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int = None,
            hidden_act: str = "relu",
            output_norm: bool = True,
            num_layers: int = 4,
            theta_dim: int = 16,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size

        layers = []
        in_dim = hidden_size
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, intermediate_size, bias=False))
            layers.append(nn.ReLU())
            in_dim = intermediate_size
        self.stack = nn.Sequential(*layers)
        self.theta_proj = nn.Linear(intermediate_size, theta_dim, bias=False)
        self.basis_proj = nn.Linear(theta_dim, hidden_size, bias=False)
        self.output_norm = ExpertRMSNorm(hidden_size) if output_norm else nn.Identity()

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stack(x)
        theta = self.theta_proj(h)
        out = self.basis_proj(theta)
        return self.output_norm(out)

    def zero_init_output(self):
        nn.init.zeros_(self.basis_proj.weight)
