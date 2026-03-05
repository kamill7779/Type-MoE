import torch
from torch import nn

from .base import BaseTokenExpert
from .common import ExpertRMSNorm
from .registry import register_expert


@register_expert("fedformer_cycle")
class FedFormerCycleExpert(BaseTokenExpert):
    expert_type = "cycle"
    interface_kind = "seq"

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int = None,
            hidden_act: str = "silu",
            output_norm: bool = True,
            modes: int = 32,
    ):
        super().__init__()
        self.modes = max(1, int(modes))
        self.freq_weight = nn.Parameter(torch.randn(self.modes, hidden_size) * 0.02)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_norm = ExpertRMSNorm(hidden_size) if output_norm else nn.Identity()

    def forward_seq(self, x: torch.Tensor, token_index=None) -> torch.Tensor:
        # x: [B, L, H]
        dtype = x.dtype
        x_f = x.to(torch.float32)
        x_fft = torch.fft.rfft(x_f, dim=1)
        n_freq = x_fft.shape[1]
        m = min(self.modes, n_freq)

        out_fft = torch.zeros_like(x_fft)
        weights = self.freq_weight[:m].to(x_fft.dtype)
        out_fft[:, :m, :] = x_fft[:, :m, :] * weights.unsqueeze(0)
        out = torch.fft.irfft(out_fft, n=x.shape[1], dim=1).to(dtype)
        out = self.out_proj(out)
        return self.output_norm(out)

    def zero_init_output(self):
        nn.init.zeros_(self.out_proj.weight)
