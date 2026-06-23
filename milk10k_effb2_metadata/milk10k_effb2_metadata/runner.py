"""Single-split and k-fold training runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from milk10k_effb2_metadata.data import (
    fit_metadata_spec,
    hybrid_balance_summary,
    kfold_splits,
    lesion_split,
    load_paired_dataframe,
    make_loaders,
    metadata_vector,
)
from milk10k_effb2_metadata.engine import train_phase
from milk10k_effb2_metadata.losses import build_loss
from milk10k_effb2_metadata.metrics import apply_class_bias, compute_metrics, optimize_class_bias, predict, save_predictions
from milk10k_effb2_metadata.model_setup import build_model, load_model_state_compat, load_resume_checkpoint
from milk10k_effb2_metadata.models import DualEffB2MetadataClassifier
from milk10k_effb2_metadata.reporting import build_data_summary, save_data_summary, save_kfold_report, save_run_diagnostics
from milk10k_effb2_metadata.training_utils import json_safe, save_kfold_summary, save_run_config

def train_lws_post_training(
    model: DualEffB2MetadataClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    source_checkpoint: dict[str, Any],
    output_path: Path,
) -> dict[str, Any] | None:
    if args.lws_epochs <= 0:
        return None

    print(f"\nStarting LWS Post-Training for {args.lws_epochs} epochs...")
    model.requires_grad_(False)
    model.class_scales.data.fill_(1.0)
    model.class_scales.requires_grad_(True)
    optimizer = torch.optim.Adam([model.class_scales], lr=args.lws_lr)
    criterion = nn.CrossEntropyLoss()

    dataset = train_loader.dataset
    labels = np.asarray(dataset.labels, dtype=np.int64)
    counts = np.bincount(labels)
    class_weights = 1.0 / np.power(counts.astype(np.float64), args.lws_sampler_power)
    generator = torch.Generator().manual_seed(args.seed)
    lws_sampler = WeightedRandomSampler(
        torch.as_tensor(class_weights[labels], dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    lws_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        sampler=lws_sampler,
    )

    # Keep dropout and batch normalization disabled. Gradients still flow to
    # class_scales while every representation/classifier parameter is frozen.
    model.eval()
    from milk10k_effb2_metadata.metrics import move_batch
    best_score = float("-inf")
    best_metrics: dict[str, Any] | None = None
    for epoch in range(1, args.lws_epochs + 1):
        total_loss = 0.0
        for batch in lws_loader:
            clinical, dermoscopic, metadata, labels = move_batch(batch, device)
            optimizer.zero_grad()
            logits = model(clinical, dermoscopic, metadata)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            model.class_scales.data.clamp_(args.lws_min_scale, args.lws_max_scale)
            total_loss += loss.item()

        y_true, y_prob = predict(model, val_loader, device)
        metrics, _, _ = compute_metrics(y_true, y_prob, source_checkpoint["class_names"])
        scales_str = np.array2string(model.class_scales.detach().cpu().numpy(), precision=3, separator=',')
        print(
            f"LWS Epoch {epoch}/{args.lws_epochs} - Loss: {total_loss / max(len(lws_loader), 1):.4f} "
            f"- F1: {metrics['f1_macro']:.4f} - Scales: {scales_str}"
        )
        if metrics[args.selection_metric] > best_score:
            best_score = float(metrics[args.selection_metric])
            best_metrics = metrics
            payload = dict(source_checkpoint)
            payload["model_state"] = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            payload["checkpoint_variant"] = "lws"
            payload["best_selection_metric"] = best_score
            payload["best_val_f1_macro"] = float(metrics["f1_macro"])
            payload["lws_epoch"] = epoch
            payload["lws_scales"] = model.class_scales.detach().cpu().tolist()
            payload["variant_val_metrics"] = json_safe(metrics)
            torch.save(payload, output_path)
    model.class_scales.requires_grad_(False)
    return best_metrics

def fit_global_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    all_logits = []
    all_labels = []
    from milk10k_effb2_metadata.metrics import move_batch
    with torch.no_grad():
        for batch in val_loader:
            clinical, dermoscopic, metadata, labels = move_batch(batch, device)
            logits = model(clinical, dermoscopic, metadata)
            all_logits.append(logits)
            all_labels.append(labels)
            
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)
    
    def eval_fn():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(all_logits / temperature, all_labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_fn)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


@torch.no_grad()
def predict_temperature(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    from milk10k_effb2_metadata.metrics import move_batch

    model.eval()
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    for batch in loader:
        clinical, dermoscopic, metadata, labels = move_batch(batch, device)
        logits = model(clinical, dermoscopic, metadata) / temperature
        labels_all.append(labels.cpu().numpy())
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def add_head_confidence_metrics(
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    train_df: pd.DataFrame,
    min_support: int = 100,
) -> None:
    train_counts = train_df["label"].value_counts()
    head_indices = [idx for idx, name in enumerate(class_names) if int(train_counts.get(name, 0)) >= min_support]
    y_pred = y_prob.argmax(axis=1)
    mask = np.isin(y_true, head_indices) & (y_pred == y_true)
    metrics["head_class_names"] = [class_names[idx] for idx in head_indices]
    metrics["mean_correct_confidence_head"] = (
        float(y_prob[mask, y_true[mask]].mean()) if np.any(mask) else None
    )


def build_tail_tracking_config(
    train_df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.loss != "ldam" or args.tail_num_classes <= 0:
        return None

    counts_series = train_df["label"].value_counts().reindex(class_names, fill_value=0)
    train_class_counts = {label: int(counts_series[label]) for label in class_names}
    tail_class_names = sorted(class_names, key=lambda label: (train_class_counts[label], label))[
        : min(args.tail_num_classes, len(class_names))
    ]
    return {
        "tail_class_names": tail_class_names,
        "tail_class_indices": [label_to_idx[label] for label in tail_class_names],
        "train_class_counts": train_class_counts,
    }


def resolve_label_name(class_names: list[str], name: str) -> str:
    normalized = {label.upper(): label for label in class_names}
    key = name.strip().upper()
    if key not in normalized:
        raise ValueError(f"Unknown augmented class name: {name!r}. Choices: {class_names}")
    return normalized[key]


def source_lesion_id(value: Any) -> str:
    """Return the original lesion ID for a generated paired lesion ID."""
    return str(value).split("__sdpair_", 1)[0]


def load_augmented_subset(
    base_df: pd.DataFrame,
    class_names: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    augmented_data_dir = getattr(args, "augmented_data_dir", None)
    if augmented_data_dir is None:
        return pd.DataFrame(columns=base_df.columns)
    augmented_dir = augmented_data_dir.expanduser().resolve()
    augmented_df = load_paired_dataframe(augmented_dir)
    base_lesion_ids = set(base_df["lesion_id"].astype(str))
    augmented_df = augmented_df[~augmented_df["lesion_id"].astype(str).isin(base_lesion_ids)].copy()
    augmented_classes = getattr(args, "augmented_classes", [])
    if augmented_classes:
        allowed = {resolve_label_name(class_names, name) for name in augmented_classes}
        augmented_df = augmented_df[augmented_df["label"].isin(allowed)].copy()
    augmented_max_per_class = getattr(args, "augmented_max_per_class", 0)
    if augmented_max_per_class < 0:
        raise ValueError("--augmented-max-per-class must be >= 0.")
    augmented_df["is_augmented"] = True
    augmented_df["ignore_metadata"] = bool(getattr(args, "zero_augmented_metadata", False))
    return augmented_df


def append_augmented_train_rows(
    base_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    augmented_df = load_augmented_subset(base_df, class_names, args)
    if augmented_df.empty:
        if getattr(args, "augmented_data_dir", None) is not None:
            print("Augmented data: no extra rows selected.")
        return train_df
    train_source_ids = set(train_df["lesion_id"].astype(str).map(source_lesion_id))
    val_source_ids = set(val_df["lesion_id"].astype(str).map(source_lesion_id))
    augmented_df["source_lesion_id"] = augmented_df["lesion_id"].astype(str).map(source_lesion_id)
    source_overlap = train_source_ids & val_source_ids
    if source_overlap:
        raise RuntimeError(
            f"Source leakage already exists between train and validation: {len(source_overlap)} lesion IDs."
        )
    selected = augmented_df["source_lesion_id"].isin(train_source_ids)
    excluded_validation = augmented_df["source_lesion_id"].isin(val_source_ids)
    unknown = ~(selected | excluded_validation)
    if unknown.any():
        examples = augmented_df.loc[unknown, "lesion_id"].astype(str).head(5).tolist()
        raise ValueError(
            "Augmented lesions cannot be mapped to an original train/validation source. "
            f"Examples: {examples}"
        )
    excluded_count = int(excluded_validation.sum())
    augmented_df = augmented_df.loc[selected].copy()
    augmented_max_per_class = getattr(args, "augmented_max_per_class", 0)
    if augmented_max_per_class > 0 and not augmented_df.empty:
        augmented_df = (
            augmented_df.sample(frac=1.0, random_state=args.seed)
            .groupby("label", group_keys=False)
            .head(augmented_max_per_class)
            .sort_values(["label", "lesion_id"])
            .reset_index(drop=True)
        )
    counts = augmented_df["label"].value_counts().sort_index().to_dict()
    print(
        "Source-safe augmented train append: "
        f"rows={len(augmented_df)}, counts={counts}, "
        f"excluded_validation_sources={excluded_count}, "
        f"zero_metadata={getattr(args, 'zero_augmented_metadata', False)}, "
        f"source={getattr(args, 'augmented_data_dir', None)}"
    )
    return pd.concat([train_df, augmented_df], ignore_index=True, sort=False)


def run_training_split(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
    output_dir: Path,
    fold: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    data_summary = build_data_summary(df, train_df, val_df, class_names)
    if args.balance_mode == "hybrid":
        data_summary["balance"] = hybrid_balance_summary(
            [label_to_idx[label] for label in train_df["label"].tolist()],
            {idx: label for label, idx in label_to_idx.items()},
            args,
        )
    save_data_summary(output_dir, data_summary)

    metadata_spec = fit_metadata_spec(train_df)
    metadata_dim = len(metadata_vector(train_df.iloc[0], metadata_spec))
    save_run_config(
        output_dir,
        args,
        class_names,
        label_to_idx,
        metadata_spec,
        train_df,
        val_df,
        clinical_backbone_backend,
        dermoscopic_backbone_backend,
        fold,
    )

    model = build_model(
        class_names,
        metadata_dim,
        args,
        device,
        clinical_backbone_backend,
        dermoscopic_backbone_backend,
    )
    
    ema_model = None
    if getattr(args, "ema", False):
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
        
    resume_epoch, resume_best_val_f1, resume_phase = load_resume_checkpoint(args.resume_checkpoint, model, device, ema_model=ema_model)
    train_loader, val_loader = make_loaders(train_df, val_df, label_to_idx, metadata_spec, args)
    criterion = build_loss(train_df, label_to_idx, args, device)
    tail_config = build_tail_tracking_config(train_df, class_names, label_to_idx, args)

    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Paired lesions: train={len(train_df)}, val={len(val_df)}, total={len(df)}")
    print(f"Metadata input dim: {metadata_dim}")
    print(f"MONET columns: {len(metadata_spec.get('monet_columns', []))}")
    print(
        f"Metadata mode: disable_metadata={args.disable_metadata}, "
        f"freeze_metadata_head={args.freeze_metadata_head}, metadata_lr={args.metadata_lr}, "
        f"metadata_fusion={args.metadata_fusion}, image_fusion={getattr(args, 'image_fusion', 'concat')}, "
        f"gate_hidden_dim={args.metadata_gate_hidden_dim}"
    )
    print(
        f"Loss: {args.loss}, class_weight={args.class_weight}, weighted_sampler={args.weighted_sampler}, "
        f"balance_mode={args.balance_mode}"
    )
    if args.balance_mode == "hybrid":
        print(f"Hybrid balance plan: {data_summary['balance']}")
    if getattr(args, "image_fusion", "concat") == "moe" and args.logit_fusion_mode == "fixed":
        print("Note: --image-fusion moe already mixes expert logits; --logit-fusion-mode fixed adds extra branch logits.")
    if args.loss == "ce_f1":
        print(f"Soft-F1 class controls: ignore={args.f1_ignore_classes}, weights={args.f1_class_weight}")
    if args.loss == "ldam" and args.class_weight:
        print("Note: --class-weight is ignored for --loss ldam because LDAM+DRW uses effective-number alpha.")
    if tail_config is not None:
        tail_counts = {label: tail_config["train_class_counts"][label] for label in tail_config["tail_class_names"]}
        print(f"LDAM tail tracking: tail_num_classes={args.tail_num_classes}, tail_counts={tail_counts}")

    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.csv"
    if args.resume_checkpoint is not None and history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")
    best_start = resume_best_val_f1 if args.resume_checkpoint is not None else float("-inf")
    best_tail_start = float("-inf")
    tail_best_path = output_dir / "tail_best.pt"
    if args.resume_checkpoint is not None and tail_best_path.exists():
        tail_checkpoint = torch.load(tail_best_path, map_location=device, weights_only=False)
        best_tail_start = float(tail_checkpoint.get("best_val_tail_recall_macro", float("-inf")))
    skip_freeze_until = resume_epoch if resume_phase == "freeze" else 1
    if resume_phase == "finetune":
        skip_freeze_until = args.freeze_epochs + 1
    skip_finetune_until = resume_epoch if resume_phase == "finetune" else 1
    variant_best = {"raw": float("-inf"), "ema": float("-inf")}
    epoch, best_val_f1, best_val_tail_recall, variant_best = train_phase(
        "freeze",
        args.freeze_epochs,
        1,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        args,
        class_names,
        label_to_idx,
        metadata_spec,
        output_dir,
        history,
        best_start,
        skip_freeze_until,
        **(tail_config or {}),
        best_val_tail_recall=best_tail_start,
        ema_model=ema_model,
        variant_best=variant_best,
    )
    epoch, best_val_f1, best_val_tail_recall, variant_best = train_phase(
        "finetune",
        args.finetune_epochs,
        epoch,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        args,
        class_names,
        label_to_idx,
        metadata_spec,
        output_dir,
        history,
        best_val_f1,
        skip_finetune_until,
        **(tail_config or {}),
        best_val_tail_recall=best_val_tail_recall,
        ema_model=ema_model,
        variant_best=variant_best,
    )

    raw_path = output_dir / "best_raw.pt"
    ema_path = output_dir / "best_ema.pt"
    if not raw_path.exists():
        raise RuntimeError(f"Training did not produce {raw_path}")
    source_path = ema_path if ema_path.exists() else raw_path
    source_checkpoint = torch.load(source_path, map_location=device, weights_only=False)
    load_model_state_compat(model, source_checkpoint["model_state"])

    lws_path = output_dir / "best_lws.pt"
    if args.lws_epochs > 0:
        train_lws_post_training(
            model,
            train_loader,
            val_loader,
            device,
            args,
            source_checkpoint,
            lws_path,
        )

    variant_paths = [raw_path]
    if ema_path.exists():
        variant_paths.append(ema_path)
    if lws_path.exists():
        variant_paths.append(lws_path)

    variant_results: dict[str, dict[str, Any]] = {}
    deployment: tuple[float, Path, dict[str, Any], np.ndarray] | None = None
    y_true: np.ndarray | None = None
    for variant_path in variant_paths:
        checkpoint = torch.load(variant_path, map_location=device, weights_only=False)
        load_model_state_compat(model, checkpoint["model_state"])
        variant = str(checkpoint.get("checkpoint_variant", variant_path.stem.removeprefix("best_")))
        uncalibrated_y_true, uncalibrated_prob = predict_temperature(model, val_loader, device, 1.0)
        uncalibrated_metrics, _, _ = compute_metrics(uncalibrated_y_true, uncalibrated_prob, class_names)
        add_head_confidence_metrics(
            uncalibrated_metrics,
            uncalibrated_y_true,
            uncalibrated_prob,
            class_names,
            train_df,
        )
        temperature = fit_global_temperature(model, val_loader, device) if args.fit_temperature else 1.0
        current_y_true, current_prob = predict_temperature(model, val_loader, device, temperature)
        current_metrics, current_per_class, current_cm = compute_metrics(current_y_true, current_prob, class_names)
        add_head_confidence_metrics(current_metrics, current_y_true, current_prob, class_names, train_df)
        checkpoint["temperature"] = temperature
        checkpoint["uncalibrated_metrics"] = json_safe(uncalibrated_metrics)
        checkpoint["temperature_metrics"] = json_safe(current_metrics)
        checkpoint["checkpoint_variant"] = variant
        torch.save(checkpoint, variant_path)
        current_per_class.to_csv(output_dir / f"per_class_metrics_{variant}.csv", index=False)
        pd.DataFrame(current_cm, index=class_names, columns=class_names).to_csv(
            output_dir / f"confusion_matrix_{variant}.csv"
        )
        variant_output = output_dir / variant
        variant_output.mkdir(exist_ok=True)
        save_predictions(val_df, current_y_true, current_prob, class_names, variant_output)
        variant_results[variant] = {
            "checkpoint": str(variant_path),
            "temperature": temperature,
            "uncalibrated_metrics": uncalibrated_metrics,
            "metrics": current_metrics,
        }
        score = float(current_metrics[args.selection_metric])
        if deployment is None or score > deployment[0]:
            deployment = (score, variant_path, checkpoint, current_prob)
            y_true = current_y_true

    if deployment is None or y_true is None:
        raise RuntimeError("No deployable raw/EMA/LWS checkpoint was produced.")
    _, deployment_path, deployment_checkpoint, y_prob = deployment
    torch.save(deployment_checkpoint, output_dir / "best.pt")
    print(
        f"Selected deployment variant={deployment_checkpoint['checkpoint_variant']} "
        f"from {deployment_path.name}, temperature={deployment_checkpoint['temperature']:.4f}"
    )

    metrics, per_class_df, cm = compute_metrics(y_true, y_prob, class_names)
    add_head_confidence_metrics(metrics, y_true, y_prob, class_names, train_df)
    metrics = {
        "best_selection_metric": float(metrics[args.selection_metric]),
        "selection_metric_name": args.selection_metric,
        "best_val_f1_macro": float(metrics["f1_macro"]),
        "checkpoint_variant": deployment_checkpoint["checkpoint_variant"],
        "temperature": deployment_checkpoint["temperature"],
        "variants": variant_results,
        **metrics,
    }
    if tail_config is not None:
        metrics["best_val_tail_recall_macro"] = float(best_val_tail_recall)
        metrics["tail_class_names"] = tail_config["tail_class_names"]
    if args.calibrate_bias:
        class_bias, calibrated_score = optimize_class_bias(
            y_true,
            y_prob,
            class_names,
            metric_name=args.calibration_metric,
            max_bias=args.calibration_max_bias,
            step=args.calibration_step,
            passes=args.calibration_passes,
        )
        calibrated_prob = apply_class_bias(y_prob, class_bias)
        calibrated_metrics, calibrated_per_class_df, calibrated_cm = compute_metrics(y_true, calibrated_prob, class_names)
        calibration_payload = {
            "metric": args.calibration_metric,
            "optimized_score": float(calibrated_score),
            "class_names": class_names,
            "class_bias": [float(item) for item in class_bias.tolist()],
            "metrics": calibrated_metrics,
        }
        with open(output_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(calibration_payload), f, indent=2)
        calibrated_per_class_df.to_csv(output_dir / "per_class_metrics_calibrated.csv", index=False)
        pd.DataFrame(calibrated_cm, index=class_names, columns=class_names).to_csv(
            output_dir / "confusion_matrix_calibrated.csv"
        )
        metrics["calibrated"] = calibrated_metrics
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, indent=2)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    save_predictions(val_df, y_true, y_prob, class_names, output_dir)
    save_run_diagnostics(
        output_dir,
        args,
        data_summary,
        metrics,
        per_class_df,
        cm,
        y_prob,
        class_names,
        fold,
    )
    print(
        f"Done: best_val_f1_macro={metrics['f1_macro']:.4f}, "
        f"val_acc={metrics['accuracy']:.4f}, balanced_acc={metrics['balanced_accuracy']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, top3={metrics['top3_accuracy']:.4f}, "
        f"auc_macro={metrics['roc_auc_macro_ovr']}"
    )
    return metrics


def train_single_run(
    df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
) -> dict[str, Any]:
    df = df.copy()
    df["is_augmented"] = False
    df["ignore_metadata"] = False
    if args.synthetic_train_only:
        synthetic_mask = df["lesion_id"].astype(str).str.contains("__sdpair_", regex=False)
        real_df = df[~synthetic_mask].copy()
        synthetic_df = df[synthetic_mask].copy()
        train_df, val_df = lesion_split(real_df, args.val_size, args.seed)
        train_sources = set(train_df["lesion_id"].astype(str))
        val_sources = set(val_df["lesion_id"].astype(str))
        synthetic_df["source_lesion_id"] = synthetic_df["lesion_id"].astype(str).map(source_lesion_id)
        unknown_sources = ~synthetic_df["source_lesion_id"].isin(train_sources | val_sources)
        if unknown_sources.any():
            examples = synthetic_df.loc[unknown_sources, "lesion_id"].astype(str).head(5).tolist()
            raise ValueError(f"Synthetic lesions have unknown source IDs. Examples: {examples}")
        safe_synthetic_df = synthetic_df[synthetic_df["source_lesion_id"].isin(train_sources)].copy()
        excluded_count = int(synthetic_df["source_lesion_id"].isin(val_sources).sum())
        train_df = pd.concat([train_df, safe_synthetic_df], ignore_index=True, sort=False)
        print(
            f"Source-safe synthetic train-only split: real_train={len(train_df) - len(safe_synthetic_df)}, "
            f"synthetic_train={len(safe_synthetic_df)}, excluded_validation_sources={excluded_count}, "
            f"val_real={len(val_df)}"
        )
    else:
        train_df, val_df = lesion_split(df, args.val_size, args.seed)
    train_df = append_augmented_train_rows(df, train_df, val_df, class_names, args)
    return run_training_split(
        df,
        train_df,
        val_df,
        class_names,
        label_to_idx,
        args,
        device,
        clinical_backbone_backend,
        dermoscopic_backbone_backend,
        args.output_dir,
    )


def train_kfold(
    df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
) -> list[dict[str, Any]]:
    df = df.copy()
    df["is_augmented"] = False
    df["ignore_metadata"] = False
    fold_metrics = []
    for fold_idx, (train_df, val_df) in enumerate(kfold_splits(df, args.k_folds, args.seed)):
        print(f"\nK-fold {fold_idx + 1}/{args.k_folds}")
        train_df = append_augmented_train_rows(df, train_df, val_df, class_names, args)
        metrics = run_training_split(
            df,
            train_df,
            val_df,
            class_names,
            label_to_idx,
            args,
            device,
            clinical_backbone_backend,
            dermoscopic_backbone_backend,
            args.output_dir / f"fold_{fold_idx:02d}",
            fold_idx,
        )
        fold_metrics.append({"fold": fold_idx, **metrics})
    save_kfold_summary(fold_metrics, args.output_dir)
    save_kfold_report(fold_metrics, args.output_dir)
    return fold_metrics
