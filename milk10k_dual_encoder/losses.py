"""Loss functions for MILK10k dual-encoder experiments."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from milk10k_dual_encoder.config import ArchitectureConfig


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def classification_loss(train_df: pd.DataFrame, label_to_idx: dict[str, int], args: argparse.Namespace, device: torch.device) -> nn.Module:
    weight = None
    if args.class_weight:
        y = np.array([label_to_idx[label] for label in train_df["label"]])
        values = compute_class_weight(class_weight="balanced", classes=np.arange(len(label_to_idx)), y=y)
        weight = torch.tensor(values, dtype=torch.float32, device=device)
    if args.focal_loss:
        return FocalLoss(weight=weight)
    return nn.CrossEntropyLoss(weight=weight)


def symmetric_info_nce(clinical_z: torch.Tensor, dermoscopic_z: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = clinical_z @ dermoscopic_z.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def siglip_loss(clinical_z: torch.Tensor, dermoscopic_z: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = clinical_z @ dermoscopic_z.T / temperature
    labels = torch.zeros_like(logits)
    labels.diagonal().fill_(1.0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def compute_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    attributes: torch.Tensor,
    criterion: nn.Module,
    arch: ArchitectureConfig,
    args: argparse.Namespace,
    phase: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls = criterion(output["logits"], labels)
    contrastive = torch.zeros((), device=labels.device)
    attr = torch.zeros((), device=labels.device)

    if arch.objective in {"multitask", "clip", "sm3"}:
        contrastive = symmetric_info_nce(output["clinical_z"], output["dermoscopic_z"], args.temperature)
    elif arch.objective == "siglip":
        contrastive = siglip_loss(output["clinical_z"], output["dermoscopic_z"], args.temperature)
    if arch.use_attributes and "attribute_logits" in output:
        attr = F.binary_cross_entropy_with_logits(output["attribute_logits"], attributes)

    if phase == "pretrain":
        loss = contrastive + (args.attribute_weight * attr if arch.use_attributes else 0.0)
    else:
        loss = cls
        if arch.objective in {"multitask", "sm3"}:
            loss = loss + args.contrastive_weight * contrastive
        if arch.use_attributes:
            loss = loss + args.attribute_weight * attr
    return loss, {
        "classification_loss": float(cls.detach().item()),
        "contrastive_loss": float(contrastive.detach().item()),
        "attribute_loss": float(attr.detach().item()),
    }
