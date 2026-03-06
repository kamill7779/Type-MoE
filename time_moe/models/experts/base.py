from typing import Optional

import torch
from torch import nn


class BaseTokenExpert(nn.Module):
    expert_type: str = "generic"
    interface_kind: str = "flat"  # flat | seq

    def __init__(self, hidden_size: int = None):
        super().__init__()
        # Register flat_fallback as a proper nn.Module for seq experts.
        # Used when context buffer is unavailable and sequence_length == 1.
        if self.interface_kind == "seq" and hidden_size is not None:
            self.flat_fallback = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.flat_fallback.weight)
        else:
            self.flat_fallback = None

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_seq(self, x: torch.Tensor, token_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, token_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.interface_kind == "seq":
            return self.forward_seq(x, token_index=token_index)
        return self.forward_flat(x)

    def forward_flat_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback for seq experts when only a single token is available
        (KV-cache autoregressive decoding) and no context buffer is present.

        This is a zero-initialised linear projection so the expert contributes
        ≈ 0 at the start of training, matching the zero-init residual
        strategy.  Subclasses that have output_norm should override to apply
        their norm.
        """
        assert self.flat_fallback is not None, (
            "flat_fallback not registered. "
            "Seq experts must pass hidden_size to BaseTokenExpert.__init__."
        )
        return self.flat_fallback(x)

    def zero_init_output(self):
        # Optional hook for experts with explicit output projection.
        return None
