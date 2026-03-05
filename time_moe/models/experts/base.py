from typing import Optional

import torch
from torch import nn


class BaseTokenExpert(nn.Module):
    expert_type: str = "generic"
    interface_kind: str = "flat"  # flat | seq

    def __init__(self):
        super().__init__()

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_seq(self, x: torch.Tensor, token_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, token_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.interface_kind == "seq":
            return self.forward_seq(x, token_index=token_index)
        return self.forward_flat(x)

    def zero_init_output(self):
        # Optional hook for experts with explicit output projection.
        return None
