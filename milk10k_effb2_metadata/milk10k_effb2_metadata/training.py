"""Training orchestration facade for the EffB2 dual metadata classifier."""

from __future__ import annotations

import argparse

from milk10k_effb2_metadata.training_utils import json_safe


def validate_balance_args(args: argparse.Namespace) -> None:
    if args.balance_mode == "hybrid" and args.weighted_sampler:
        raise ValueError("--balance-mode hybrid cannot be combined with --weighted-sampler.")
    if args.balance_head_ratio <= 0:
        raise ValueError("--balance-head-ratio must be greater than 0.")
    if args.balance_tail_floor < 0:
        raise ValueError("--balance-tail-floor must be >= 0.")
    if args.balance_min_source_count < 1:
        raise ValueError("--balance-min-source-count must be at least 1.")
    tau = float(getattr(args, "tau", 0.0))
    class_weight = bool(getattr(args, "class_weight", False))
    loss = str(getattr(args, "loss", "ce"))
    lws_epochs = int(getattr(args, "lws_epochs", 0))
    lws_lr = float(getattr(args, "lws_lr", 1e-2))
    lws_sampler_power = float(getattr(args, "lws_sampler_power", 0.5))
    lws_min_scale = float(getattr(args, "lws_min_scale", 0.75))
    lws_max_scale = float(getattr(args, "lws_max_scale", 1.5))
    ema_decay = float(getattr(args, "ema_decay", 0.999))
    if not 0.0 <= tau <= 0.5:
        raise ValueError("--tau must be between 0.0 and 0.5.")
    if tau > 0.0 and class_weight:
        raise ValueError("--tau > 0 cannot be combined with --class-weight.")
    if tau > 0.0 and loss in {"focal", "ldam"}:
        raise ValueError("--tau > 0 requires a CE-based loss (ce, ce_dice, or ce_f1).")
    if lws_epochs < 0:
        raise ValueError("--lws-epochs must be >= 0.")
    if lws_lr <= 0.0:
        raise ValueError("--lws-lr must be > 0.")
    if not 0.0 <= lws_sampler_power <= 1.0:
        raise ValueError("--lws-sampler-power must be between 0.0 and 1.0.")
    if lws_min_scale <= 0.0 or lws_max_scale < lws_min_scale:
        raise ValueError("LWS scale bounds must satisfy 0 < min <= max.")
    if not 0.0 < ema_decay < 1.0:
        raise ValueError("--ema-decay must be between 0 and 1.")


def run(args: argparse.Namespace) -> None:
    import torch

    from datasets import resolve_data_dir, set_seed
    from milk10k_effb2_metadata.data import (
        audit_dermoscopic_masks,
        load_paired_dataframe,
        print_mask_audit_summary,
    )
    from milk10k_effb2_metadata.model_setup import resolve_training_backbone_backends
    from milk10k_effb2_metadata.models import is_one_encoder_image_fusion, normalize_backbone_name, resolve_image_size
    from milk10k_effb2_metadata.runner import train_kfold, train_single_run

    if args.k_folds < 1:
        raise ValueError("--k-folds must be at least 1.")
    validate_balance_args(args)

    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.backbone = normalize_backbone_name(args.backbone)
    if args.metadata_gate_hidden_dim is None:
        args.metadata_gate_hidden_dim = args.metadata_dim
    if is_one_encoder_image_fusion(getattr(args, "image_fusion", "concat")):
        if args.clinical_checkpoint is not None or args.dermoscopic_checkpoint is not None:
            raise ValueError(
                f"--image-fusion {args.image_fusion} uses one ImageNet-initialized encoder; "
                "do not pass --clinical-checkpoint or --dermoscopic-checkpoint."
            )
    if args.resume_checkpoint is None and args.clinical_checkpoint is None and args.dermoscopic_checkpoint is None:
        args.imagenet_pretrained = True
    args.image_size = resolve_image_size(args.backbone, args.image_size)

    df = load_paired_dataframe(data_dir)
    if not 0.0 <= args.min_dermoscopic_mask_ratio <= 1.0:
        raise ValueError("--min-dermoscopic-mask-ratio must be between 0 and 1.")
    if args.dermoscopic_mask_dir is not None:
        args.dermoscopic_mask_dir = args.dermoscopic_mask_dir.expanduser().resolve()
        df, mask_audit = audit_dermoscopic_masks(
            df,
            args.dermoscopic_mask_dir,
            args.min_dermoscopic_mask_ratio,
        )
        mask_audit.to_csv(args.output_dir / "dermoscopic_mask_audit.csv", index=False)
        print_mask_audit_summary(mask_audit, args.min_dermoscopic_mask_ratio)
    class_names = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clinical_backbone_backend, dermoscopic_backbone_backend = resolve_training_backbone_backends(args, device)

    print(f"Data dir: {data_dir}")
    if args.k_folds == 1:
        train_single_run(
            df,
            class_names,
            label_to_idx,
            args,
            device,
            clinical_backbone_backend,
            dermoscopic_backbone_backend,
        )
    else:
        train_kfold(
            df,
            class_names,
            label_to_idx,
            args,
            device,
            clinical_backbone_backend,
            dermoscopic_backbone_backend,
        )
