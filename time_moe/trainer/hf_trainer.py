#!/usr/bin/env python
# -*- coding:utf-8 _*-
import math
import logging
from dataclasses import field, dataclass
from functools import partial
from typing import List, Optional, Set

import inspect

import transformers
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler, TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class TimeMoeTrainer(transformers.Trainer):
    epsilon = 1e-8

    def __init__(self, label_column: str = 'labels', loss_mask_column: str = 'loss_mask', *positional_args, **kwargs):
        super().__init__(*positional_args, **kwargs)
        self.tokenizer = kwargs.get("tokenizer", None)
        self.label_column = label_column
        self.loss_mask_column = loss_mask_column

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        optimizer = self.optimizer if optimizer is None else optimizer
        min_lr_ratio = self.args.min_learning_rate / self.args.learning_rate
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_type == 'cosine':
                self.lr_scheduler = get_cosine_schedule_with_warmup_min_lr(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=min_lr_ratio,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            params = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns = list(set(
                params + self.label_names + [
                    "label",
                    "label_ids",
                    self.label_column,
                    self.loss_mask_column
                ]
            ))


@dataclass
class TimeMoETrainingArguments(transformers.TrainingArguments):
    min_learning_rate: float = field(
        default=0, metadata={"help": "Minimum learning rate for cosine_schedule"}
    )


def _get_cosine_schedule_with_warmup_and_min_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

    return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_ratio)


def get_cosine_schedule_with_warmup_min_lr(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0,
        last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_and_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ---------------------------------------------------------------------------
# Freeze-strategy callbacks (plan §10.1)
# ---------------------------------------------------------------------------

def _identify_new_expert_indices(config) -> Set[int]:
    """Return expert indices marked with ``zero_init_output: true`` in
    ``custom_expert_specs``, i.e. newly-added (non-pretrained) experts."""
    specs = getattr(config, "custom_expert_specs", None) or []
    indices = set()
    for i, spec in enumerate(specs):
        if isinstance(spec, dict) and spec.get("zero_init_output", False):
            indices.add(i)
    return indices


def _param_name_matches(name: str, patterns: List[str]) -> bool:
    """Return True if ``name`` contains any of the given substring patterns."""
    for p in patterns:
        if p in name:
            return True
    return False


def _set_requires_grad(model: torch.nn.Module, name_patterns: List[str], value: bool):
    """Set ``requires_grad`` for parameters whose name contains any of the
    given patterns."""
    for name, param in model.named_parameters():
        if _param_name_matches(name, name_patterns):
            param.requires_grad = value


def _freeze_all(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_all(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True


class PhasedFreezeCallback(TrainerCallback):
    """Implements the 3-phase unfreeze protocol from plan §10.1.

    - Phase-A (step 0 → ``phase_a_end``): only gate weights are trainable.
    - Phase-B (step ``phase_a_end`` → ``phase_b_end``): gate + new experts.
    - Phase-C (step ``phase_b_end`` +): everything is trainable.

    "New experts" are those whose ``custom_expert_specs`` entry has
    ``zero_init_output: true``.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            config,
            phase_a_end: int = 1000,
            phase_b_end: int = 5000,
    ):
        super().__init__()
        self.phase_a_end = phase_a_end
        self.phase_b_end = phase_b_end
        self._current_phase: Optional[str] = None

        # Build name-match patterns for each group
        new_expert_indices = _identify_new_expert_indices(config)
        self._gate_patterns = [".gate."]  # matches ffn_layer.gate.weight etc.
        self._new_expert_patterns = [f".experts.{i}." for i in new_expert_indices]

        # Apply Phase-A immediately
        self._apply_phase_a(model)

    def _apply_phase_a(self, model: torch.nn.Module):
        if self._current_phase == "A":
            return
        _freeze_all(model)
        _set_requires_grad(model, self._gate_patterns, True)
        self._current_phase = "A"
        logger.info("PhasedFreezeCallback: entered Phase-A (gate only)")

    def _apply_phase_b(self, model: torch.nn.Module):
        if self._current_phase == "B":
            return
        _freeze_all(model)
        _set_requires_grad(model, self._gate_patterns + self._new_expert_patterns, True)
        self._current_phase = "B"
        logger.info("PhasedFreezeCallback: entered Phase-B (gate + new experts)")

    def _apply_phase_c(self, model: torch.nn.Module):
        if self._current_phase == "C":
            return
        _unfreeze_all(model)
        self._current_phase = "C"
        logger.info("PhasedFreezeCallback: entered Phase-C (all params)")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, model=None, **kwargs):
        if model is None:
            return
        step = state.global_step
        if step < self.phase_a_end:
            self._apply_phase_a(model)
        elif step < self.phase_b_end:
            self._apply_phase_b(model)
        else:
            self._apply_phase_c(model)


class GateOnlyFreezeCallback(TrainerCallback):
    """Permanently freeze everything except gate weights."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        _freeze_all(model)
        _set_requires_grad(model, [".gate."], True)
        logger.info("GateOnlyFreezeCallback: only gate params are trainable")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                      control: TrainerControl, model=None, **kwargs):
        # No state transitions — gate_only is permanent.
        pass
