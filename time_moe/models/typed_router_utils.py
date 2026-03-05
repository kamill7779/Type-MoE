"""
Typed routing utilities for Type-MoE.

Contains typed_preselect, RouterInfo, load balancing loss,
type diversity loss, and routing statistics helpers.

Extracted from modeling_time_moe.py per plan §13.2.
"""
import warnings
from collections import namedtuple
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F

from transformers.utils import logging

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# RouterInfo — per-layer routing metadata that travels through the forward chain.
# Replaces the plain ``router_logits`` tensor so that loss functions can consume
# pre-computed topk results directly instead of re-computing the route.
# ---------------------------------------------------------------------------
RouterInfo = namedtuple("RouterInfo", [
    "raw_logits",        # [T, N] — raw gate logits (before softmax, after jitter)
    "topk_indices",      # [T, k] — indices of selected experts after typed_preselect + topk
    "topk_weights",      # [T, k] — corresponding routing weights (float, pre-cast)
    "filtered_probs",    # [T, N] — softmax probs after typed_preselect (zeros for non-selected)
    "raw_probs",         # [T, N] — softmax probs *before* typed_preselect (all experts keep their probs)
    "actual_k",          # int — the actual k used (may be < requested top_k)
])


# ---------------------------------------------------------------------------
# typed_preselect
# ---------------------------------------------------------------------------
def typed_preselect(
        routing_weights: torch.Tensor,
        expert_type_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Keep only one winner expert per type for each token.

    Args:
        routing_weights: [T, N_experts] softmax probabilities.
        expert_type_ids: [N_experts] integer type id per expert.

    Returns:
        filtered: [T, N_experts] with at most ``num_types`` non-zero entries per row.
    """
    if expert_type_ids is None:
        return routing_weights
    if expert_type_ids.numel() != routing_weights.shape[-1]:
        raise ValueError(
            f"expert_type_ids size mismatch: expected {routing_weights.shape[-1]}, "
            f"got {expert_type_ids.numel()}"
        )

    filtered = torch.zeros_like(routing_weights)
    num_types = int(expert_type_ids.max().item()) + 1
    token_indices = torch.arange(routing_weights.size(0), device=routing_weights.device)
    for type_id in range(num_types):
        type_mask = (expert_type_ids == type_id)
        if not torch.any(type_mask):
            continue
        type_probs = routing_weights[:, type_mask]
        best_local = type_probs.argmax(dim=-1)
        global_indices = type_mask.nonzero(as_tuple=True)[0]
        best_global = global_indices[best_local]
        filtered[token_indices, best_global] = routing_weights[token_indices, best_global]
    return filtered


# ---------------------------------------------------------------------------
# _resolve_actual_k — with stability fix
# ---------------------------------------------------------------------------
def _resolve_actual_k(filtered_weights: torch.Tensor, requested_top_k: int) -> int:
    """Determine safe top-k given filtered probabilities.

    If some tokens have fewer non-zero candidates than ``requested_top_k``,
    the returned k is clamped to the minimum valid count across all tokens.
    A warning is emitted when clamping occurs.
    """
    valid_counts = (filtered_weights > 0).sum(dim=-1)
    if valid_counts.numel() == 0:
        return 1
    min_valid = int(valid_counts.min().item())
    if min_valid <= 0:
        # This can happen if a token has all-zero filtered weights.
        # Warn and fallback to k=1, preserving the least-bad behaviour.
        logger.warning_once(
            "typed_preselect produced tokens with zero valid experts; "
            "falling back to actual_k=1. Check expert_type_map configuration."
        )
        return 1
    actual = max(1, min(int(requested_top_k), min_valid))
    if actual < requested_top_k:
        logger.warning_once(
            f"_resolve_actual_k: clamped top_k from {requested_top_k} to {actual} "
            f"because some tokens only have {min_valid} non-zero candidates after typed_preselect."
        )
    return actual


# ---------------------------------------------------------------------------
# _collect_routing_stats — **FIX**: P_i uses raw (pre-filtered) probs
# ---------------------------------------------------------------------------
def _collect_routing_stats(
        raw_routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        num_experts: int,
        token_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-expert token assignment fraction (f_i) and mean routing
    probability (P_i).

    **Key design decision (plan §4.3):** ``P_i`` is computed from the
    **pre-filtered** ``raw_routing_weights`` (original softmax output before
    ``typed_preselect``).  This ensures that every expert — including those
    eliminated by typed_preselect — still receives gradient signal through the
    aux loss, preventing the Matthew effect within type groups.

    ``f_i`` is computed from ``selected_experts`` (the post-filtered topk
    result), which correctly reflects actual token assignments.
    """
    expert_mask = F.one_hot(selected_experts, num_experts).float()
    if token_mask is None:
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) / float(expert_mask.shape[0] * expert_mask.shape[1])
        router_prob_per_expert = torch.mean(raw_routing_weights, dim=0)
        return tokens_per_expert, router_prob_per_expert

    token_mask = token_mask.to(raw_routing_weights.device).float()
    expanded_mask = token_mask[:, None, None]
    denom_assign = (expanded_mask.sum() * selected_experts.shape[1]).clamp(min=1.0)
    tokens_per_expert = (expert_mask * expanded_mask).sum(dim=(0, 1)) / denom_assign

    denom_prob = token_mask.sum().clamp(min=1.0)
    router_prob_per_expert = (raw_routing_weights * token_mask[:, None]).sum(dim=0) / denom_prob
    return tokens_per_expert, router_prob_per_expert


# ---------------------------------------------------------------------------
# Helpers to unpack RouterInfo from the legacy or new format
# ---------------------------------------------------------------------------
def _unpack_layer_info(layer_gate):
    """Return (raw_logits, topk_indices, topk_weights, raw_probs) from either
    a plain tensor (legacy) or a RouterInfo namedtuple."""
    if isinstance(layer_gate, RouterInfo):
        return layer_gate.raw_logits, layer_gate.topk_indices, layer_gate.topk_weights, layer_gate.raw_probs
    # Legacy path: plain tensor — caller must re-compute routing
    return layer_gate, None, None, None


# ---------------------------------------------------------------------------
# load_balancing_loss_func
# ---------------------------------------------------------------------------
def load_balancing_loss_func(
        gate_logits: Union[torch.Tensor, Tuple, List],
        top_k: int,
        num_experts: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        router_mode: str = "standard",
        expert_type_map: Optional[List[int]] = None,
) -> torch.Tensor:
    r"""Computes auxiliary load balancing loss as in Switch Transformer.

    When ``gate_logits`` elements are :class:`RouterInfo` namedtuples the
    pre-computed topk results are reused directly (avoiding re-computation
    and the jitter-mismatch issue).  Otherwise falls back to the legacy
    path that re-computes softmax + topk from raw logits.
    """
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)):
        return torch.tensor(0.0)
    # The first element may be None when the layer doesn't have a router (dense layer).
    first = gate_logits[0]
    if first is None:
        return torch.tensor(0.0)

    compute_device = first.raw_logits.device if isinstance(first, RouterInfo) else first.device
    if num_experts is None:
        num_experts = first.raw_logits.shape[-1] if isinstance(first, RouterInfo) else first.shape[-1]

    typed_mode = (router_mode == "typed_topk"
                  and expert_type_map is not None
                  and len(expert_type_map) == num_experts)
    if typed_mode:
        expert_type_ids = torch.tensor(expert_type_map, dtype=torch.long, device=compute_device)
    else:
        expert_type_ids = None

    token_mask = None
    if attention_mask is not None:
        token_mask = attention_mask.reshape(-1).to(compute_device).float()

    layer_losses = []
    for layer_gate in gate_logits:
        raw_logits, pre_topk_indices, pre_topk_weights, pre_raw_probs = _unpack_layer_info(layer_gate)
        raw_logits = raw_logits.to(compute_device)

        if pre_topk_indices is not None:
            # ---- NEW path: reuse forward's routing decisions ----
            selected_experts = pre_topk_indices.to(compute_device)
            raw_probs = pre_raw_probs.to(compute_device)
        else:
            # ---- LEGACY path: recompute (no jitter, for backward compat) ----
            routing_weights = F.softmax(raw_logits, dim=-1)
            raw_probs = routing_weights
            filtered_weights = typed_preselect(routing_weights, expert_type_ids) if typed_mode else routing_weights
            layer_k = _resolve_actual_k(filtered_weights, top_k)
            _, selected_experts = torch.topk(filtered_weights, layer_k, dim=-1)

        layer_mask = token_mask
        if layer_mask is not None and layer_mask.numel() != raw_logits.shape[0]:
            layer_mask = None

        tokens_per_expert, router_prob_per_expert = _collect_routing_stats(
            raw_routing_weights=raw_probs,
            selected_experts=selected_experts,
            num_experts=num_experts,
            token_mask=layer_mask,
        )
        layer_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
        layer_losses.append(layer_loss * num_experts)

    if len(layer_losses) == 0:
        return torch.tensor(0.0, device=compute_device)
    return torch.stack(layer_losses).mean()


# ---------------------------------------------------------------------------
# type_diversity_loss_func
# ---------------------------------------------------------------------------
def type_diversity_loss_func(
        gate_logits: Union[torch.Tensor, Tuple, List],
        top_k: int,
        expert_type_map: Optional[List[int]],
        attention_mask: Optional[torch.Tensor] = None,
        router_mode: str = "standard",
) -> torch.Tensor:
    """Encourage top-k selections to cover diverse expert types."""
    if gate_logits is None or not isinstance(gate_logits, (tuple, list)):
        return torch.tensor(0.0)
    first = gate_logits[0]
    if first is None:
        return torch.tensor(0.0)

    if top_k <= 1 or router_mode != "typed_topk":
        compute_device = first.raw_logits.device if isinstance(first, RouterInfo) else first.device
        return torch.tensor(0.0, device=compute_device)

    compute_device = first.raw_logits.device if isinstance(first, RouterInfo) else first.device
    num_experts_total = first.raw_logits.shape[-1] if isinstance(first, RouterInfo) else first.shape[-1]
    if expert_type_map is None or len(expert_type_map) != num_experts_total:
        return torch.tensor(0.0, device=compute_device)

    expert_type_ids = torch.tensor(expert_type_map, dtype=torch.long, device=compute_device)
    num_types = int(expert_type_ids.max().item()) + 1

    token_mask = None
    if attention_mask is not None:
        token_mask = attention_mask.reshape(-1).to(compute_device).bool()

    losses = []
    for layer_gate in gate_logits:
        raw_logits, pre_topk_indices, _, _ = _unpack_layer_info(layer_gate)

        if pre_topk_indices is not None:
            selected_experts = pre_topk_indices.to(compute_device)
            layer_k = selected_experts.shape[-1]
        else:
            raw_logits = raw_logits.to(compute_device)
            routing_weights = F.softmax(raw_logits, dim=-1)
            filtered_weights = typed_preselect(routing_weights, expert_type_ids)
            layer_k = _resolve_actual_k(filtered_weights, top_k)
            _, selected_experts = torch.topk(filtered_weights, layer_k, dim=-1)

        if token_mask is not None and token_mask.numel() == selected_experts.shape[0]:
            selected_experts = selected_experts[token_mask]
        if selected_experts.numel() == 0:
            continue

        selected_types = expert_type_ids[selected_experts]
        type_one_hot = F.one_hot(selected_types, num_classes=num_types).float()
        unique_per_token = (type_one_hot.sum(dim=1) > 0).sum(dim=-1).float()
        target_unique = float(min(layer_k, num_types))
        losses.append((target_unique - unique_per_token).clamp(min=0).mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=compute_device)
    return torch.stack(losses).mean()
