import math

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseTokenExpert
from .common import ExpertRMSNorm
from .registry import register_expert


@register_expert("anomaly_attn")
class AnomalyTokenExpert(BaseTokenExpert):
    expert_type = "anomaly"
    interface_kind = "seq"

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int = None,
            hidden_act: str = "silu",
            output_norm: bool = True,
            num_heads: int = 8,
            causal: bool = True,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigma_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_norm = ExpertRMSNorm(hidden_size) if output_norm else nn.Identity()

    def _gaussian_prior(self, sigma: torch.Tensor, length: int) -> torch.Tensor:
        # sigma: [B, heads, L]
        pos = torch.arange(length, device=sigma.device, dtype=sigma.dtype)
        dist = (pos[:, None] - pos[None, :]).abs()  # [L, L]

        sigma = torch.sigmoid(sigma * 5.0) + 1e-5
        sigma = torch.pow(torch.tensor(3.0, device=sigma.device, dtype=sigma.dtype), sigma) - 1.0
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, length)  # [B, heads, L, L]
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-(dist[None, None, :, :] ** 2) / (2.0 * sigma ** 2))
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return prior

    def forward_seq(self, x: torch.Tensor, token_index=None) -> torch.Tensor:
        # x: [B, L, H]
        B, L, H = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        sigma = self.sigma_proj(x).transpose(1, 2)  # [B, heads, L]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
        series = F.softmax(scores, dim=-1)
        prior = self._gaussian_prior(sigma=sigma, length=L).to(series.dtype)

        mixed_attn = 0.5 * series + 0.5 * prior
        out = torch.matmul(mixed_attn, v).transpose(1, 2).reshape(B, L, H)
        out = self.out_proj(out)
        return self.output_norm(out)

    def forward_flat_fallback(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward_flat_fallback(x)
        return self.output_norm(out)

    def zero_init_output(self):
        nn.init.zeros_(self.out_proj.weight)
