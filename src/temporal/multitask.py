from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MultitaskOutput:
    return_prediction: torch.Tensor
    direction_logits: torch.Tensor


@dataclass(frozen=True)
class MultitaskLossResult:
    total_loss: torch.Tensor
    return_loss: torch.Tensor
    direction_loss: torch.Tensor


def returns_to_direction_classes(
    returns: torch.Tensor,
    flat_threshold: float = 0.005,
) -> torch.Tensor:
    classes = torch.ones_like(returns, dtype=torch.long)
    classes = torch.where(returns < -flat_threshold, torch.zeros_like(classes), classes)
    classes = torch.where(returns > flat_threshold, torch.full_like(classes, 2), classes)
    return classes


def multitask_loss(
    return_prediction: torch.Tensor,
    direction_logits: torch.Tensor,
    return_targets: torch.Tensor,
    direction_targets: torch.Tensor,
    regression_loss_fn: nn.Module | None = None,
    direction_loss_fn: nn.Module | None = None,
    direction_loss_weight: float = 0.5,
) -> MultitaskLossResult:
    regression_loss_fn = regression_loss_fn or nn.SmoothL1Loss()
    direction_loss_fn = direction_loss_fn or nn.CrossEntropyLoss()
    return_loss = regression_loss_fn(return_prediction, return_targets)
    direction_loss = direction_loss_fn(direction_logits, direction_targets)
    total_loss = return_loss + direction_loss_weight * direction_loss
    return MultitaskLossResult(
        total_loss=total_loss,
        return_loss=return_loss,
        direction_loss=direction_loss,
    )


def inverse_direction_classes(classes: torch.Tensor) -> torch.Tensor:
    return classes.to(torch.long) - 1
