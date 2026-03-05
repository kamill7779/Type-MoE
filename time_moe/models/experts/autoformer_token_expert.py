import torch
from torch import nn

from .base import BaseTokenExpert
from .common import ExpertRMSNorm
from .registry import register_expert


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        tail = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, tail], dim=1)
        x = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return trend, seasonal


class SimplifiedAutoCorrelation(nn.Module):
    def __init__(self, top_k_freq: int = 3):
        super().__init__()
        self.top_k_freq = top_k_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        dtype = x.dtype
        x_f = x.to(torch.float32)
        x_fft = torch.fft.rfft(x_f, dim=1)
        power = x_fft.abs().mean(dim=(0, 2))
        k = max(1, min(self.top_k_freq, power.shape[0]))
        top_idx = torch.topk(power, k=k, dim=-1).indices

        out_fft = torch.zeros_like(x_fft)
        out_fft[:, top_idx, :] = x_fft[:, top_idx, :]
        out = torch.fft.irfft(out_fft, n=x.shape[1], dim=1)
        return out.to(dtype)


@register_expert("autoformer_trend")
class AutoformerTrendExpert(BaseTokenExpert):
    expert_type = "trend"
    interface_kind = "seq"

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int = None,
            hidden_act: str = "silu",
            output_norm: bool = True,
            kernel_size: int = 25,
    ):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=kernel_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_norm = ExpertRMSNorm(hidden_size) if output_norm else nn.Identity()

    def forward_seq(self, x: torch.Tensor, token_index=None) -> torch.Tensor:
        trend, _ = self.decomp(x)
        out = self.out_proj(trend)
        return self.output_norm(out)

    def zero_init_output(self):
        nn.init.zeros_(self.out_proj.weight)


@register_expert("autoformer_cycle")
class AutoformerCycleExpert(BaseTokenExpert):
    expert_type = "cycle"
    interface_kind = "seq"

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int = None,
            hidden_act: str = "silu",
            output_norm: bool = True,
            kernel_size: int = 25,
            top_k_freq: int = 3,
    ):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size=kernel_size)
        self.auto_corr = SimplifiedAutoCorrelation(top_k_freq=top_k_freq)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_norm = ExpertRMSNorm(hidden_size) if output_norm else nn.Identity()

    def forward_seq(self, x: torch.Tensor, token_index=None) -> torch.Tensor:
        _, seasonal = self.decomp(x)
        out = self.auto_corr(seasonal)
        out = self.out_proj(out)
        return self.output_norm(out)

    def zero_init_output(self):
        nn.init.zeros_(self.out_proj.weight)
