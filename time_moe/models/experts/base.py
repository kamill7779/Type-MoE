from typing import Optional

import torch
from torch import nn


class BaseTokenExpert(nn.Module):
    expert_type: str = "generic"
    interface_kind: str = "flat"  # flat | seq

    def __init__(self):
        super().__init__()
        # Lazy-initialised flat fallback projection for seq experts at KV-cache
        # inference time (sequence_length == 1).
        self._flat_fallback: Optional[nn.Linear] = None

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
        (KV-cache autoregressive decoding).

        By default this is a zero-initialised linear projection so the expert
        contributes ≈ 0 at the start of training, matching the zero-init
        residual strategy used for new experts.  Subclasses may override.
        """
        if self._flat_fallback is None:
            hidden_size = x.shape[-1]
            self._flat_fallback = nn.Linear(hidden_size, hidden_size, bias=False).to(
                device=x.device, dtype=x.dtype
            )
            nn.init.zeros_(self._flat_fallback.weight)
        return self._flat_fallback(x)

    def zero_init_output(self):
        # Optional hook for experts with explicit output projection.
        return None
