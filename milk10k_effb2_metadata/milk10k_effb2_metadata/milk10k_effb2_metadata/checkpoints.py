"""Checkpoint loading utilities for mixed timm/torchvision EfficientNet-B2 branches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_STATE_KEYS = ("model_state", "model_state_dict", "state_dict")
PREFIXES_TO_STRIP = ("module.", "model.", "_orig_mod.")


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in CHECKPOINT_STATE_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint does not contain a supported state dict.")


def load_raw_checkpoint(path: Path, device: torch.device, branch_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{branch_name} checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def normalize_key(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES_TO_STRIP:
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                changed = True
    return key


def infer_checkpoint_backend(path: Path, device: torch.device, branch_name: str) -> str:
    checkpoint = load_raw_checkpoint(path, device, branch_name)
    state = extract_state_dict(checkpoint)
    keys = {normalize_key(key) for key in state}
    timm_prefixes = ("conv_stem.", "bn1.", "blocks.", "conv_head.", "bn2.", "stages.", "stem.")
    torchvision_prefixes = ("features.", "avgpool.", "classifier.")
    timm_hits = sum(key.startswith(timm_prefixes) for key in keys)
    torchvision_hits = sum(key.startswith(torchvision_prefixes) for key in keys)
    if timm_hits > torchvision_hits:
        return "timm"
    if torchvision_hits > timm_hits:
        return "torchvision"
    if any(key.startswith("layer") for key in keys):
        return "timm"
    raise RuntimeError(
        f"{branch_name}: cannot infer checkpoint backend from {path}. "
        "Pass --backbone-backend timm or --backbone-backend torchvision explicitly."
    )


def resolve_backbone_backends(args: argparse.Namespace, device: torch.device) -> tuple[str, str]:
    if args.backbone_backend != "auto":
        return args.backbone_backend, args.backbone_backend

    clinical_backend = infer_checkpoint_backend(args.clinical_checkpoint, device, "clinical")
    dermoscopic_backend = infer_checkpoint_backend(args.dermoscopic_checkpoint, device, "dermoscopic")
    print(f"Auto-detected backbone backends: clinical={clinical_backend}, dermoscopic={dermoscopic_backend}")
    return clinical_backend, dermoscopic_backend


def load_encoder_checkpoint(path: Path, encoder: nn.Module, branch_name: str, device: torch.device) -> None:
    checkpoint = load_raw_checkpoint(path, device, branch_name)
    raw_state = extract_state_dict(checkpoint)
    source_state = {normalize_key(key): value for key, value in raw_state.items()}
    target_state = encoder.state_dict()
    matched = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    skipped = len(source_state) - len(matched)
    if not matched:
        raise RuntimeError(f"{branch_name}: no matching encoder weights loaded from {path}")

    target_state.update(matched)
    encoder.load_state_dict(target_state)
    print(f"{branch_name}: loaded {len(matched)} keys from {path}; skipped {skipped} keys")

