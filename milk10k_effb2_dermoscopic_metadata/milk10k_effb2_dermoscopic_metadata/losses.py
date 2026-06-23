"""Classification losses for the dermoscopic-only trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__(); self.weight = weight; self.gamma = gamma

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        loss = (1.0 - torch.exp(-ce)) ** self.gamma * ce
        if self.weight is not None: loss = loss * self.weight[labels]
        return loss.mean()


class LDAMLoss(nn.Module):
    def __init__(self, class_counts, beta=.9999, max_margin=.5, deferred_start_epoch=0, alpha_max=10.0):
        super().__init__()
        counts = class_counts.float().clamp_min(1)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts)); margins *= max_margin / margins.max().clamp_min(1e-12)
        alpha = effective_number_alpha(counts, beta).clamp(max=alpha_max); alpha *= counts.numel() / alpha.sum().clamp_min(1e-12)
        self.register_buffer("margins", margins); self.register_buffer("alpha", alpha)
        self.deferred_start_epoch = deferred_start_epoch; self.current_epoch = 0

    def set_epoch(self, epoch): self.current_epoch = epoch

    def forward(self, logits, labels):
        adjusted = logits.clone(); rows = torch.arange(labels.size(0), device=labels.device)
        adjusted[rows, labels] -= self.margins.to(logits)[labels]
        loss = F.cross_entropy(adjusted, labels, reduction="none")
        if self.current_epoch >= self.deferred_start_epoch: loss *= self.alpha.to(logits)[labels]
        return loss.mean()


class SoftMacroDiceLoss(nn.Module):
    def forward(self, logits, labels):
        probs = torch.softmax(logits, 1); target = F.one_hot(labels, logits.size(1)).to(probs.dtype)
        score = (2 * (probs * target).sum(0) + 1e-6) / (probs.sum(0) + target.sum(0) + 1e-6)
        return 1 - score.mean()


class SoftMacroF1Loss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None: self.register_buffer("class_weights", class_weights.float())
        else: self.class_weights = None

    def forward(self, logits, labels):
        probs = torch.softmax(logits, 1); target = F.one_hot(labels, logits.size(1)).to(probs.dtype)
        tp = (probs * target).sum(0); fp = (probs * (1-target)).sum(0); fn = ((1-probs) * target).sum(0)
        f1 = (2*tp + 1e-6) / (2*tp + fp + fn + 1e-6)
        if self.class_weights is None: return 1 - f1.mean()
        weights = self.class_weights.to(probs); return 1 - (f1 * weights).sum() / weights.sum().clamp_min(1e-6)


class CompositeClassificationLoss(nn.Module):
    def __init__(self, primary, auxiliary, weight):
        super().__init__(); self.primary = primary; self.auxiliary = auxiliary; self.weight = weight
    def forward(self, logits, labels): return self.primary(logits, labels) + self.weight * self.auxiliary(logits, labels)


def effective_number_alpha(counts, beta):
    if beta <= 0: return torch.ones_like(counts)
    if beta >= 1: raise ValueError("--ldam-beta must be less than 1")
    b = torch.tensor(beta, dtype=counts.dtype, device=counts.device)
    return (1-b) / (1-torch.pow(b, counts)).clamp_min(1e-12)


def class_counts(train_df, label_to_idx, device):
    values = np.bincount([label_to_idx[x] for x in train_df.label], minlength=len(label_to_idx))
    if np.any(values == 0): raise ValueError("Train split contains an empty class.")
    return torch.tensor(values, dtype=torch.float32, device=device)


def f1_weights(label_to_idx, args, device):
    weights = torch.ones(len(label_to_idx), device=device)
    normalized = {name.upper(): name for name in label_to_idx}
    for name in args.f1_ignore_classes:
        key = name.upper();
        if key not in normalized: raise ValueError(f"Unknown F1 class: {name}")
        weights[label_to_idx[normalized[key]]] = 0
    for item in args.f1_class_weight:
        if "=" not in item: raise ValueError(f"Expected CLASS=VALUE: {item}")
        name, raw = item.split("=", 1); value = float(raw)
        if value < 0 or name.upper() not in normalized: raise ValueError(f"Invalid F1 class weight: {item}")
        weights[label_to_idx[normalized[name.upper()]]] = value
    if weights.sum() <= 0: raise ValueError("F1 class weights sum to zero.")
    return weights


def build_loss(train_df: pd.DataFrame, label_to_idx, args, device):
    if args.loss == "ldam":
        return LDAMLoss(class_counts(train_df, label_to_idx, device), args.ldam_beta, args.ldam_max_margin,
                        args.ldam_drw_start_epoch, args.ldam_alpha_max)
    weight = None
    if args.class_weight:
        y = np.array([label_to_idx[x] for x in train_df.label])
        weight = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(label_to_idx)), y=y), dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weight)
    if args.loss == "focal": return FocalLoss(weight, args.focal_gamma)
    if args.loss == "ce_dice": return CompositeClassificationLoss(ce, SoftMacroDiceLoss(), args.dice_weight)
    if args.loss == "ce_f1": return CompositeClassificationLoss(ce, SoftMacroF1Loss(f1_weights(label_to_idx,args,device)), args.f1_weight)
    return ce
