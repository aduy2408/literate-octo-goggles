"""Model, optimizer, and checkpoint setup for metadata training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from milk10k_effb2_metadata.checkpoints import (
    infer_checkpoint_backend,
    load_encoder_checkpoint,
    resolve_backbone_backends,
)
from milk10k_effb2_metadata.models import (
    DualEffB2MetadataClassifier,
    is_one_encoder_image_fusion,
    model_class_for_backbone,
)


def load_model_state_compat(model: DualEffB2MetadataClassifier, state: dict[str, torch.Tensor]) -> None:
    """Load checkpoints created before LWS added the class_scales parameter."""
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {"class_scales"}
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if "class_scales" in missing:
        model.class_scales.data.fill_(1.0)


def infer_branch_backend_from_state(state: dict[str, torch.Tensor], branch_prefix: str) -> str:
    keys = [key.removeprefix(branch_prefix) for key in state if key.startswith(branch_prefix)]
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
    raise RuntimeError(f"Cannot infer backend for resume checkpoint branch prefix {branch_prefix!r}.")


def resolve_training_backbone_backends(args: argparse.Namespace, device: torch.device) -> tuple[str, str]:
    image_fusion = getattr(args, "image_fusion", "concat")
    if is_one_encoder_image_fusion(image_fusion):
        if args.backbone_backend != "auto":
            return args.backbone_backend, args.backbone_backend
        if args.resume_checkpoint is not None:
            checkpoint = torch.load(args.resume_checkpoint.expanduser().resolve(), map_location=device, weights_only=False)
            state = checkpoint["model_state"]
            backend = infer_branch_backend_from_state(state, "shared_encoder.")
            checkpoint_args = checkpoint.get("args", {})
            if checkpoint_args.get("backbone") and args.backbone == "efficientnet_b2":
                args.backbone = checkpoint_args["backbone"]
            print(f"Auto-detected resume shared backend: shared={backend}")
            return backend, backend
        print("One-encoder image fusion: using timm backbone initialized from ImageNet weights.")
        return "timm", "timm"
    if args.backbone_backend != "auto":
        return args.backbone_backend, args.backbone_backend
    if args.clinical_checkpoint is not None and args.dermoscopic_checkpoint is not None:
        return resolve_backbone_backends(args, device)
    if args.clinical_checkpoint is not None:
        clinical_backend = infer_checkpoint_backend(args.clinical_checkpoint, device, "clinical")
        print(
            "Auto-detected clinical backbone backend: "
            f"clinical={clinical_backend}, dermoscopic={clinical_backend} (ImageNet initialized)"
        )
        return clinical_backend, clinical_backend
    if args.dermoscopic_checkpoint is not None:
        dermoscopic_backend = infer_checkpoint_backend(args.dermoscopic_checkpoint, device, "dermoscopic")
        print(
            "Auto-detected dermoscopic backbone backend: "
            f"clinical={dermoscopic_backend} (ImageNet initialized), dermoscopic={dermoscopic_backend}"
        )
        return dermoscopic_backend, dermoscopic_backend
    if args.resume_checkpoint is None:
        print("No branch checkpoints passed; using timm backbones initialized from ImageNet weights.")
        return "timm", "timm"
    checkpoint = torch.load(args.resume_checkpoint.expanduser().resolve(), map_location=device, weights_only=False)
    state = checkpoint["model_state"]
    clinical_backend = infer_branch_backend_from_state(state, "clinical_encoder.")
    dermoscopic_backend = infer_branch_backend_from_state(state, "dermoscopic_encoder.")
    checkpoint_args = checkpoint.get("args", {})
    if checkpoint_args.get("backbone") and args.backbone == "efficientnet_b2":
        args.backbone = checkpoint_args["backbone"]
    print(f"Auto-detected resume backends: clinical={clinical_backend}, dermoscopic={dermoscopic_backend}")
    return clinical_backend, dermoscopic_backend


def build_optimizer(
    model: DualEffB2MetadataClassifier,
    args: argparse.Namespace,
    encoders_trainable: bool,
) -> torch.optim.Optimizer:
    head_params = []
    encoder_params = []
    metadata_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("clinical_encoder.", "dermoscopic_encoder.", "shared_encoder.")):
            encoder_params.append(param)
        elif name.startswith(
            (
                "metadata_head.",
                "clinical_metadata_gate.",
                "dermoscopic_metadata_gate.",
                "shared_metadata_gate.",
            )
        ):
            metadata_params.append(param)
        else:
            head_params.append(param)

    groups = [{"params": head_params, "lr": args.head_lr}]
    if metadata_params:
        groups.append({"params": metadata_params, "lr": args.metadata_lr if args.metadata_lr is not None else args.head_lr})
    if encoders_trainable and encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def set_metadata_head_trainable(model: DualEffB2MetadataClassifier, trainable: bool) -> None:
    for param in model.metadata_head.parameters():
        param.requires_grad = trainable
    for module_name in ("clinical_metadata_gate", "dermoscopic_metadata_gate", "shared_metadata_gate"):
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = trainable


def load_resume_checkpoint(
    checkpoint_path: Path | None,
    model: DualEffB2MetadataClassifier,
    device: torch.device,
    ema_model: torch.nn.Module | None = None,
) -> tuple[int, float, str | None]:
    if checkpoint_path is None:
        return 1, float("-inf"), None
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_model_state_compat(model, checkpoint["model_state"])
    if ema_model is not None and "ema_model_state" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state"])
    next_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_f1 = float(
        checkpoint.get(
            "best_selection_metric",
            checkpoint.get("best_val_f1_macro", float("-inf")),
        )
    )
    phase = checkpoint.get("phase")
    selection_metric_name = checkpoint.get("selection_metric_name", "f1_macro")
    print(
        f"Resumed checkpoint: {checkpoint_path}, phase={phase}, "
        f"last_epoch={next_epoch - 1}, best_{selection_metric_name}={best_val_f1:.4f}"
    )
    print("Optimizer is re-created from current CLI LR settings.")
    return next_epoch, best_val_f1, str(phase) if phase is not None else None


def build_model(
    class_names: list[str],
    metadata_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
) -> DualEffB2MetadataClassifier:
    model_class = model_class_for_backbone(args.backbone)
    model = model_class(
        num_classes=len(class_names),
        metadata_input_dim=metadata_dim,
        branch_dim=args.branch_dim,
        metadata_dim=args.metadata_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        imagenet_pretrained=args.imagenet_pretrained,
        clinical_backbone_backend=clinical_backbone_backend,
        dermoscopic_backbone_backend=dermoscopic_backbone_backend,
        backbone=args.backbone,
        disable_metadata=args.disable_metadata,
        metadata_fusion=args.metadata_fusion,
        image_fusion=getattr(args, "image_fusion", "concat"),
        metadata_gate_hidden_dim=args.metadata_gate_hidden_dim,
        classifier_style=getattr(args, "classifier_style", "legacy"),
        logit_fusion_mode=args.logit_fusion_mode,
        fusion_logit_weight=args.fusion_logit_weight,
        clinical_logit_weight=args.clinical_logit_weight,
        dermoscopic_logit_weight=args.dermoscopic_logit_weight,
    ).to(device)
    if args.resume_checkpoint is None and not is_one_encoder_image_fusion(getattr(args, "image_fusion", "concat")):
        if args.clinical_checkpoint is not None:
            load_encoder_checkpoint(args.clinical_checkpoint, model.clinical_encoder, "clinical", device)
        if args.dermoscopic_checkpoint is not None:
            load_encoder_checkpoint(args.dermoscopic_checkpoint, model.dermoscopic_encoder, "dermoscopic", device)
    if args.disable_metadata or args.freeze_metadata_head:
        set_metadata_head_trainable(model, False)
    return model
