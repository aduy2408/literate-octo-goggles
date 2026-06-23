"""Classification losses for the EffB2 metadata trainer."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.weight is not None:
            loss = loss * self.weight[labels]
        return loss.mean()


class GeneralizedBalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        class_counts: torch.Tensor,
        tau: float = 1.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= tau <= 0.5:
            raise ValueError("--tau must be between 0.0 and 0.5.")
        if weight is not None:
            raise ValueError("Generalized Balanced Softmax cannot be combined with class weights.")
        self.tau = tau
        counts = class_counts.float().clamp_min(1.0)
        self.register_buffer("log_counts", torch.log(counts))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.training and self.tau > 0.0:
            log_counts = self.log_counts.to(device=logits.device, dtype=logits.dtype)
            adjusted_logits = logits + self.tau * log_counts
        else:
            adjusted_logits = logits
        return F.cross_entropy(adjusted_logits, labels)


class LDAMLoss(nn.Module):
    """LDAM with deferred effective-number reweighting."""

    def __init__(
        self,
        class_counts: torch.Tensor,
        beta: float = 0.9999,
        max_margin: float = 0.5,
        deferred_start_epoch: int = 0,
        alpha_max: float = 10.0,
    ) -> None:
        super().__init__()
        counts = class_counts.float().clamp_min(1.0)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (max_margin / margins.max().clamp_min(1e-12))
        alpha = effective_number_alpha(counts, beta)
        alpha = alpha.clamp(max=alpha_max)
        alpha = alpha * (counts.numel() / alpha.sum().clamp_min(1e-12))

        self.register_buffer("margins", margins)
        self.register_buffer("alpha", alpha)
        self.deferred_start_epoch = deferred_start_epoch
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        margins = self.margins.to(device=logits.device, dtype=logits.dtype)
        alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
        adjusted_logits = logits.clone()
        rows = torch.arange(labels.size(0), device=labels.device)
        adjusted_logits[rows, labels] = adjusted_logits[rows, labels] - margins[labels]
        loss = F.cross_entropy(adjusted_logits, labels, reduction="none")
        if self.current_epoch >= self.deferred_start_epoch:
            loss = loss * alpha[labels]
        return loss.mean()


class SoftMacroDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).to(dtype=probs.dtype)
        intersection = (probs * one_hot).sum(dim=0)
        denominator = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class SoftMacroF1Loss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.eps = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).to(dtype=probs.dtype)
        tp = (probs * one_hot).sum(dim=0)
        fp = (probs * (1.0 - one_hot)).sum(dim=0)
        fn = ((1.0 - probs) * one_hot).sum(dim=0)
        f1 = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
        if self.class_weights is None:
            return 1.0 - f1.mean()
        weights = self.class_weights.to(device=logits.device, dtype=probs.dtype)
        if weights.numel() != logits.size(1):
            raise RuntimeError(f"Expected {logits.size(1)} F1 class weights, got {weights.numel()}.")
        denominator = weights.sum().clamp_min(self.eps)
        return 1.0 - (f1 * weights).sum() / denominator


class CompositeClassificationLoss(nn.Module):
    def __init__(self, ce_loss: nn.Module, auxiliary_loss: nn.Module, auxiliary_weight: float) -> None:
        super().__init__()
        self.ce_loss = ce_loss
        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_weight = auxiliary_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(logits, labels) + self.auxiliary_weight * self.auxiliary_loss(logits, labels)


def effective_number_alpha(counts: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0.0:
        return torch.ones_like(counts)
    if beta >= 1.0:
        raise ValueError("--ldam-beta must be less than 1.0")
    beta_tensor = torch.tensor(beta, dtype=counts.dtype, device=counts.device)
    effective_num = 1.0 - torch.pow(beta_tensor, counts)
    alpha = (1.0 - beta_tensor) / effective_num.clamp_min(1e-12)
    return alpha


def class_count_tensor(train_df: pd.DataFrame, label_to_idx: dict[str, int], device: torch.device) -> torch.Tensor:
    y = np.array([label_to_idx[label] for label in train_df["label"]])
    counts = np.bincount(y, minlength=len(label_to_idx))
    if np.any(counts == 0):
        missing = [label for label, idx in label_to_idx.items() if counts[idx] == 0]
        raise ValueError(f"Cannot build loss because train split has zero samples for classes: {missing}")
    return torch.tensor(counts, dtype=torch.float32, device=device)


def resolve_label_name(label_to_idx: dict[str, int], name: str) -> str:
    normalized = {label.upper(): label for label in label_to_idx}
    key = name.strip().upper()
    if key not in normalized:
        raise ValueError(f"Unknown class name for F1 loss: {name!r}. Choices: {sorted(label_to_idx)}")
    return normalized[key]


def f1_class_weight_tensor(label_to_idx: dict[str, int], args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    weights = torch.ones(len(label_to_idx), dtype=torch.float32, device=device)
    for class_name in getattr(args, "f1_ignore_classes", []):
        resolved = resolve_label_name(label_to_idx, class_name)
        weights[label_to_idx[resolved]] = 0.0
    for item in getattr(args, "f1_class_weight", []):
        if "=" not in item:
            raise ValueError(f"--f1-class-weight expects CLASS=VALUE, got {item!r}.")
        class_name, value = item.split("=", 1)
        resolved = resolve_label_name(label_to_idx, class_name)
        weight = float(value)
        if weight < 0.0:
            raise ValueError(f"--f1-class-weight must be non-negative, got {item!r}.")
        weights[label_to_idx[resolved]] = weight
    if float(weights.sum().item()) <= 0.0:
        raise ValueError("F1 class weights sum to zero. Keep at least one class active for --loss ce_f1.")
    return weights


def build_loss(train_df: pd.DataFrame, label_to_idx: dict[str, int], args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.loss == "ldam":
        counts = class_count_tensor(train_df, label_to_idx, device)
        return LDAMLoss(
            class_counts=counts,
            beta=args.ldam_beta,
            max_margin=args.ldam_max_margin,
            deferred_start_epoch=args.ldam_drw_start_epoch,
            alpha_max=args.ldam_alpha_max,
        )

    weight = None
    if args.class_weight:
        y = np.array([label_to_idx[label] for label in train_df["label"]])
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(label_to_idx)), y=y)
        weight = torch.tensor(weights, dtype=torch.float32, device=device)
        
    if getattr(args, "tau", 0.0) > 0.0:
        if args.class_weight:
            raise ValueError("--tau > 0 cannot be combined with --class-weight.")
        counts = class_count_tensor(train_df, label_to_idx, device)
        ce_loss: nn.Module = GeneralizedBalancedSoftmaxLoss(counts, tau=args.tau)
    else:
        ce_loss: nn.Module = nn.CrossEntropyLoss(weight=weight)
    if args.loss == "focal":
        return FocalLoss(weight=weight, gamma=args.focal_gamma)
    if args.loss == "ce_dice":
        return CompositeClassificationLoss(ce_loss, SoftMacroDiceLoss(), args.dice_weight)
    if args.loss == "ce_f1":
        return CompositeClassificationLoss(ce_loss, SoftMacroF1Loss(f1_class_weight_tensor(label_to_idx, args, device)), args.f1_weight)
    return ce_loss
