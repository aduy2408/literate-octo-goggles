"""Retrieval and prediction metrics for paired MILK10k encoders."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def retrieval_metrics(clinical_z: torch.Tensor, dermoscopic_z: torch.Tensor) -> dict[str, float]:
    sim = clinical_z @ dermoscopic_z.T
    n = sim.size(0)

    def direction_metrics(scores: torch.Tensor, prefix: str) -> dict[str, float]:
        target = torch.arange(scores.size(0), device=scores.device)
        order = scores.argsort(dim=1, descending=True)
        ranks = (order == target[:, None]).nonzero()[:, 1] + 1
        return {
            f"{prefix}_recall_at_1": float((ranks <= 1).float().mean().item()),
            f"{prefix}_recall_at_5": float((ranks <= min(5, n)).float().mean().item()),
            f"{prefix}_recall_at_10": float((ranks <= min(10, n)).float().mean().item()),
            f"{prefix}_mrr": float((1.0 / ranks.float()).mean().item()),
            f"{prefix}_median_rank": float(ranks.float().median().item()),
        }

    return {**direction_metrics(sim, "clinical_to_derm"), **direction_metrics(sim.T, "derm_to_clinical")}


def unpack_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    metadata = batch["metadata"].to(device, non_blocking=True)
    attributes = batch["attributes"].to(device, non_blocking=True)
    return images, labels, metadata, attributes


@torch.no_grad()
def predict_and_embed(model, loader: DataLoader, device: torch.device):
    model.eval()
    labels_all = []
    probs_all = []
    clinical_z_all = []
    dermoscopic_z_all = []
    for batch in tqdm(loader, leave=False):
        images, labels, metadata, _ = unpack_batch(batch, device)
        output = model(images, metadata)
        labels_all.append(labels.cpu().numpy())
        probs_all.append(torch.softmax(output["logits"], dim=1).cpu().numpy())
        clinical_z_all.append(output["clinical_z"].cpu())
        dermoscopic_z_all.append(output["dermoscopic_z"].cpu())
    return (
        np.concatenate(labels_all),
        np.concatenate(probs_all),
        torch.cat(clinical_z_all, dim=0),
        torch.cat(dermoscopic_z_all, dim=0),
    )


def normalized_retrieval_metrics(clinical_z: torch.Tensor, dermoscopic_z: torch.Tensor) -> dict[str, float]:
    return retrieval_metrics(F.normalize(clinical_z, dim=1), F.normalize(dermoscopic_z, dim=1))
