"""Training and evaluation orchestration for MILK10k architecture runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from timm.utils import ModelEmaV2
from tqdm.auto import tqdm

from datasets import lesion_level_train_val_split, resolve_data_dir, set_seed
from milk10k_dual_encoder.config import ARCHITECTURES, MODEL_SPECS
from milk10k_dual_encoder.data import (
    fit_metadata_spec,
    load_paired_dataframe_with_extras,
    make_loaders,
    metadata_vector,
)
from milk10k_dual_encoder.losses import classification_loss, compute_loss
from milk10k_dual_encoder.metrics import normalized_retrieval_metrics, predict_and_embed, unpack_batch
from milk10k_dual_encoder.models import LightweightDualEncoderModel, set_encoder_trainable
from milk10k_dual_encoder.INDIVIDUAL_SCRIPTS.train_milk10k_baselines import compute_classification_metrics, load_checkpoint, save_checkpoint


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
    head_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name and "metadata_encoder" not in name:
            encoder_params.append(param)
        else:
            head_params.append(param)
    groups = [{"params": head_params, "lr": args.lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def run_epoch(model, loader, criterion, arch, args, device, optimizer=None, scaler=None, phase="finetune"):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "classification_loss": 0.0, "contrastive_loss": 0.0, "attribute_loss": 0.0}
    correct = 0
    top3_correct = 0
    total = 0
    use_amp = args.amp and device.type == "cuda"

    for batch in tqdm(loader, leave=False):
        images, labels, metadata, attributes = unpack_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast("cuda", enabled=use_amp):
                output = model(images, metadata)
                loss, loss_parts = compute_loss(output, labels, attributes, criterion, arch, args, phase)
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
            if training and args.ema:
                # Update EMA model
                if hasattr(model, "ema_model"):
                    model.ema_model.update(model)

        batch_size = labels.size(0)
        totals["loss"] += float(loss.detach().item()) * batch_size
        for key, value in loss_parts.items():
            totals[key] += value * batch_size
        logits = output["logits"]
        correct += (logits.argmax(dim=1) == labels).sum().item()
        topk = min(3, logits.size(1))
        top3_correct += logits.topk(topk, dim=1).indices.eq(labels[:, None]).any(dim=1).sum().item()
        total += batch_size

    stats = {key: value / max(total, 1) for key, value in totals.items()}
    stats["accuracy"] = correct / max(total, 1)
    stats["top3_accuracy"] = top3_correct / max(total, 1)
    return stats


def evaluate_and_save(model, loader, eval_df, class_names, run_dir, run_name, device, criterion, arch, args) -> dict[str, Any]:
    eval_stats = run_epoch(model, loader, criterion, arch, args, device)
    y_true, y_prob, clinical_z, dermoscopic_z = predict_and_embed(model, loader, device)
    y_pred = y_prob.argmax(axis=1)
    cls_metrics, per_class_df, cm = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    metrics = {f"val_{key}": float(value) for key, value in eval_stats.items()}
    metrics.update(cls_metrics)
    retrieval = normalized_retrieval_metrics(clinical_z, dermoscopic_z)

    with open(run_dir / f"{run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / f"{run_name}_retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=2)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(run_dir / f"{run_name}_confusion_matrix.csv")
    per_class_df.to_csv(run_dir / f"{run_name}_per_class_metrics.csv", index=False)
    _save_predictions(eval_df, y_true, y_pred, y_prob, class_names, run_dir, run_name)
    return {"classification": metrics, "retrieval": retrieval}


def _save_predictions(eval_df, y_true, y_pred, y_prob, class_names, run_dir, run_name) -> None:
    prediction_df = pd.DataFrame(
        {
            "lesion_id": eval_df["lesion_id"].tolist(),
            "clinical_path": eval_df["clinical_path"].tolist(),
            "dermoscopic_path": eval_df["dermoscopic_path"].tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
            "label_true": [class_names[i] for i in y_true],
            "label_pred": [class_names[i] for i in y_pred],
            "confidence": y_prob.max(axis=1),
        }
    )
    probability_df = pd.DataFrame(y_prob, columns=[f"prob_{name}" for name in class_names])
    pd.concat([prediction_df, probability_df], axis=1).to_csv(run_dir / f"{run_name}_val_predictions.csv", index=False)


def load_paired_augmentation_manifest(manifest_path: Path, class_names: list[str]) -> pd.DataFrame:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Paired augmentation manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {
        "class_name",
        "synthetic_lesion_id",
        "clinical_generated_path",
        "dermoscopic_generated_path",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Paired augmentation manifest is missing columns: {missing}")

    unknown = sorted(set(manifest["class_name"].dropna().astype(str)) - set(class_names))
    if unknown:
        raise ValueError(f"Paired augmentation manifest contains unknown classes: {unknown}")

    augmented = pd.DataFrame(
        {
            "lesion_id": manifest["synthetic_lesion_id"].astype(str),
            "label": manifest["class_name"].astype(str),
            "clinical_path": manifest["clinical_generated_path"].astype(str),
            "dermoscopic_path": manifest["dermoscopic_generated_path"].astype(str),
            "is_synthetic": True,
        }
    )
    for column in ("clinical_path", "dermoscopic_path"):
        missing_paths = [path for path in augmented[column].tolist() if not Path(path).exists()]
        if missing_paths:
            preview = ", ".join(missing_paths[:3])
            raise FileNotFoundError(f"{column} has {len(missing_paths)} missing synthetic images. Examples: {preview}")
    return augmented


def train_lws_post_training(model, train_loader, val_loader, class_names, device, args):
    print(f"Starting LWS post-training for {args.lws_epochs} epochs...")
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze class_scales
    if hasattr(model, "class_scales"):
        model.class_scales.requires_grad = True
    else:
        print("Model does not have class_scales parameter.")
        return
    
    # We use a standard CE loss (no logit adjustment during LWS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([model.class_scales], lr=args.lr * 10)
    
    best_val_loss = float("inf")
    best_scales = model.class_scales.data.clone()
    
    for epoch in range(1, args.lws_epochs + 1):
        model.train()
        for batch in tqdm(train_loader, leave=False, desc="LWS"):
            images, labels, metadata, _ = unpack_batch(batch, device)
            optimizer.zero_grad()
            output = model(images, metadata)
            loss = criterion(output["logits"], labels)
            loss.backward()
            optimizer.step()
            
            # PGD clamping
            model.class_scales.data.clamp_(0.75, 1.5)
            
        # Eval
        model.eval()
        val_loss = 0.0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels, metadata, _ = unpack_batch(batch, device)
                output = model(images, metadata)
                loss = criterion(output["logits"], labels)
                val_loss += loss.item() * labels.size(0)
                total += labels.size(0)
        val_loss /= total
        print(f"LWS Epoch {epoch}: val_loss={val_loss:.4f}, scales={model.class_scales.data.cpu().numpy()}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_scales = model.class_scales.data.clone()
            
    model.class_scales.data.copy_(best_scales)
    print("Finished LWS post-training.")

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, logits):
        return logits / self.temperature

def fit_global_temperature(model, val_loader, device):
    print("Fitting Global Temperature Scaling...")
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            images, labels, metadata, _ = unpack_batch(batch, device)
            output = model(images, metadata)
            all_logits.append(output["logits"])
            all_labels.append(labels)
            
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    
    def eval_fn():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(scaler(all_logits), all_labels)
        loss.backward()
        return loss
        
    optimizer.step(eval_fn)
    print(f"Optimal temperature: {scaler.temperature.item():.4f}")
    return scaler.temperature.item()

def train_architecture(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    arch = ARCHITECTURES[args.architecture]
    data_dir = resolve_data_dir(args.data_dir)
    timm_name, default_size = MODEL_SPECS[args.model]
    image_size = args.image_size or default_size
    run_name = f"{arch.key}_{args.model}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "splits").mkdir(exist_ok=True)

    df = load_paired_dataframe_with_extras(data_dir)
    df["is_synthetic"] = False
    class_names = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    train_df, val_df = lesion_level_train_val_split(df, args.val_size, args.seed)
    if args.paired_augmentation_manifest is not None:
        if arch.use_metadata:
            raise ValueError("--paired-augmentation-manifest is intended for image-only architectures, not metadata_fusion.")
        augmented_df = load_paired_augmentation_manifest(args.paired_augmentation_manifest, class_names)
        train_df = pd.concat([train_df, augmented_df], ignore_index=True, sort=False)
    train_df.to_csv(run_dir / "splits" / "train.csv", index=False)
    val_df.to_csv(run_dir / "splits" / "val.csv", index=False)
    metadata_spec = fit_metadata_spec(train_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata_dim = len(metadata_vector(train_df.iloc[0], metadata_spec)) if arch.use_metadata else 0
    model = LightweightDualEncoderModel(
        timm_name=timm_name,
        num_classes=len(class_names),
        arch=arch,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        metadata_input_dim=metadata_dim,
        metadata_dim=args.metadata_dim,
        pretrained=not args.no_pretrained,
    ).to(device)
    if args.freeze_encoders:
        set_encoder_trainable(model, False)

    train_loader, val_loader = make_loaders(train_df, val_df, label_to_idx, model, image_size, metadata_spec, args)
    criterion = classification_loss(train_df, label_to_idx, args, device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    checkpoint_path = run_dir / f"{run_name}_best.pt"
    start_epoch, best_val_loss = _maybe_resume(args, checkpoint_path, model, optimizer, device)
    
    if args.ema:
        model.ema_model = ModelEmaV2(model, decay=args.ema_decay)

    print(f"Architecture: {arch.key}, model={args.model}, timm={timm_name}, image_size={image_size}, device={device}")
    print(f"Paired lesions: train={len(train_df)}, val={len(val_df)}, classes={class_names}")
    history = []
    for phase, total_epochs in _phase_lengths(args, arch):
        if total_epochs <= 0:
            continue
        best_val_loss, start_epoch = _run_phase(
            phase, start_epoch, total_epochs, model, train_loader, val_loader, criterion,
            arch, args, device, optimizer, scaler, scheduler, run_dir, run_name,
            checkpoint_path, class_names, best_val_loss, history,
        )

    if not checkpoint_path.exists():
        save_checkpoint(checkpoint_path, model, optimizer, args.epochs, best_val_loss, class_names, args)
    load_checkpoint(checkpoint_path, model, None, device)
    
    if hasattr(args, "lws_epochs") and args.lws_epochs > 0:
        train_lws_post_training(model, train_loader, val_loader, class_names, device, args)
        # Optionally save LWS scaled model
        torch.save(model.state_dict(), run_dir / f"{run_name}_lws_best.pt")
        
    # Fit temperature scaler
    temperature = fit_global_temperature(model, val_loader, device)
    # Note: the evaluate_and_save will evaluate the unscaled logits natively unless we wrap it
    # We'll just output the optimal temperature for the ensemble stage, as probability metrics are calculated there
    
    results = evaluate_and_save(model, val_loader, val_df, class_names, run_dir, run_name, device, criterion, arch, args)
    results["optimal_temperature"] = temperature
    _save_run_config(run_dir, run_name, args, arch, metadata_spec)
    return results


def _maybe_resume(args, checkpoint_path, model, optimizer, device) -> tuple[int, float]:
    if args.resume and checkpoint_path.exists():
        return load_checkpoint(checkpoint_path, model, optimizer, device)
    return 1, float("inf")


def _phase_lengths(args, arch):
    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else (args.epochs if arch.two_stage else 0)
    finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else args.epochs
    return (("pretrain", pretrain_epochs), ("finetune", finetune_epochs))


def _run_phase(
    phase, start_epoch, total_epochs, model, train_loader, val_loader, criterion, arch, args,
    device, optimizer, scaler, scheduler, run_dir, run_name, checkpoint_path, class_names,
    best_val_loss, history,
):
    patience_count = 0
    for epoch in range(start_epoch, total_epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, arch, args, device, optimizer, scaler, phase)
        val_stats = run_epoch(model, val_loader, criterion, arch, args, device, phase=phase)
        scheduler.step(val_stats["loss"])
        _append_history(history, run_dir, run_name, phase, epoch, train_stats, val_stats)
        print(
            f"{run_name} {phase} epoch {epoch:03d}: "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f}"
        )
        if phase == "finetune" and val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_count = 0
            # Save normal model
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss, class_names, args)
            # Save EMA model if enabled
            if args.ema:
                ema_path = run_dir / f"{run_name}_ema_best.pt"
                torch.save(model.ema_model.module.state_dict(), ema_path)
        elif phase == "finetune":
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping {run_name} at epoch {epoch}")
                break
    return best_val_loss, 1


def _append_history(history, run_dir, run_name, phase, epoch, train_stats, val_stats) -> None:
    row = {
        "phase": phase,
        "epoch": epoch,
        **{f"train_{key}": value for key, value in train_stats.items()},
        **{f"val_{key}": value for key, value in val_stats.items()},
    }
    history.append(row)
    pd.DataFrame(history).to_csv(run_dir / f"{run_name}_history.csv", index=False)


def _save_run_config(run_dir, run_name, args, arch, metadata_spec) -> None:
    args_dict = {key: _json_safe(value) for key, value in vars(args).items()}
    payload = {"args": args_dict, "architecture": arch.__dict__, "metadata_spec": _json_safe(metadata_spec)}
    with open(run_dir / f"{run_name}_run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
