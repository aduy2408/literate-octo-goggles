"""CLI for the dual-backbone metadata trainer."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MILK10k dual-image backbones with metadata fusion.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--dermoscopic-mask-dir",
        type=Path,
        default=None,
        help="Optional directory containing <lesion_id>_dermoscopic_mask.png files.",
    )
    parser.add_argument(
        "--min-dermoscopic-mask-ratio",
        type=float,
        default=0.01,
        help="Fallback to the original dermoscopic image when mask foreground ratio is below this value.",
    )
    parser.add_argument(
        "--clinical-checkpoint",
        type=Path,
        default=None,
        help="Optional clinical encoder checkpoint. If omitted, the clinical branch uses ImageNet-pretrained backbone weights.",
    )
    parser.add_argument(
        "--dermoscopic-checkpoint",
        type=Path,
        default=None,
        help="Optional dermoscopic encoder checkpoint. If omitted, the dermoscopic branch uses ImageNet-pretrained backbone weights.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model weights/best score from a metadata checkpoint, usually output-dir/best.pt.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("milk10k_dual_effb2_metadata_runs"))
    parser.add_argument("--freeze-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None, help="Input image size. Defaults to backbone-specific optimal size if None.")
    parser.add_argument(
        "--backbone",
        default="efficientnet_b2",
        help=(
            "Backbone model architecture (efficientnet_b2, tf_efficientnetv2_b2, "
            "efficientnet_b1, resnet50, convnext_base)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 in small Docker/Marimo containers to avoid /dev/shm exhaustion.",
    )
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument(
        "--metadata-lr",
        type=float,
        default=None,
        help="Optional LR for metadata_head and metadata gates. Defaults to --head-lr.",
    )
    parser.add_argument(
        "--metadata-fusion",
        choices=["concat", "gated_concat", "gated_only"],
        default="concat",
        help="Metadata fusion mode. concat keeps the baseline; gated modes use metadata for channel gating.",
    )
    parser.add_argument(
        "--image-fusion",
        choices=[
            "concat",
            "cross_attention",
            "co_attention",
            "compact_bilinear",
            "low_rank_bilinear",
            "adaptive_gate",
            "moe",
            "shared_private",
            "single_encoder_canvas",
            "shared_encoder_pool",
        ],
        default="concat",
        help="Image representation fusion mode. concat keeps the baseline final fusion.",
    )
    parser.add_argument(
        "--metadata-gate-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for metadata channel gates. Defaults to --metadata-dim.",
    )
    parser.add_argument(
        "--disable-metadata",
        action="store_true",
        help="Ignore metadata values by feeding zero metadata representation and all-one metadata gates.",
    )
    parser.add_argument(
        "--freeze-metadata-head",
        action="store_true",
        help="Freeze metadata_head and metadata gate parameters while still using their current outputs.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument(
        "--synthetic-train-only",
        action="store_true",
        help="Keep synthetic lesion IDs containing __sdpair_ in train only; validation is split from real lesions.",
    )
    parser.add_argument(
        "--augmented-data-dir",
        type=Path,
        default=None,
        help="Optional augmented MILK10k-style data dir. Only extra lesion IDs are appended to the train split.",
    )
    parser.add_argument(
        "--augmented-max-per-class",
        type=int,
        default=0,
        help="Cap extra augmented lesions per class. 0 keeps all extra rows from --augmented-data-dir.",
    )
    parser.add_argument(
        "--augmented-classes",
        nargs="*",
        default=[],
        help="Optional class-name allowlist for appended augmented lesions, e.g. --augmented-classes INF BEN_OTH DF VASC.",
    )
    parser.add_argument(
        "--zero-augmented-metadata",
        action="store_true",
        help="Set metadata vectors to all zeros for rows appended from --augmented-data-dir.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--branch-dim", type=int, default=512)
    parser.add_argument("--metadata-dim", type=int, default=64)
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument(
        "--classifier-style",
        choices=["legacy", "simple"],
        default="legacy",
        help=(
            "Final fused classifier architecture. legacy keeps the existing LayerNorm/GELU head; "
            "simple uses Linear-ReLU-Dropout-Linear."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--logit-fusion-mode",
        choices=["single", "fixed"],
        default="single",
        help="single uses one fused classifier. fixed adds clinical/dermoscopic logits and mixes them with fixed weights.",
    )
    parser.add_argument("--fusion-logit-weight", type=float, default=0.6)
    parser.add_argument("--clinical-logit-weight", type=float, default=0.2)
    parser.add_argument("--dermoscopic-logit-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--sampler-power", type=float, default=1.0)
    parser.add_argument(
        "--balance-mode",
        choices=["none", "hybrid"],
        default="none",
        help="Train-only epoch balancing. hybrid caps the largest class and mildly oversamples eligible tail classes.",
    )
    parser.add_argument(
        "--balance-head-ratio",
        type=float,
        default=2.0,
        help="In hybrid mode, cap the largest class at this multiple of the second-largest class.",
    )
    parser.add_argument(
        "--balance-tail-floor",
        type=int,
        default=100,
        help="In hybrid mode, oversample eligible classes below this count up to this many rows per epoch.",
    )
    parser.add_argument(
        "--balance-min-source-count",
        type=int,
        default=20,
        help="Do not oversample a class with fewer real train rows than this value.",
    )
    parser.add_argument("--loss", choices=["ce", "focal", "ldam", "ce_dice", "ce_f1"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--f1-weight", type=float, default=0.3)
    parser.add_argument(
        "--f1-ignore-classes",
        nargs="*",
        default=[],
        help="Class names excluded from the soft macro-F1 auxiliary term, e.g. --f1-ignore-classes MAL_OTH.",
    )
    parser.add_argument(
        "--f1-class-weight",
        action="append",
        default=[],
        help="Optional CLASS=VALUE override for the soft macro-F1 auxiliary term. Can be passed multiple times.",
    )
    parser.add_argument("--ldam-beta", type=float, default=0.9999)
    parser.add_argument("--ldam-max-margin", type=float, default=0.5)
    parser.add_argument("--ldam-drw-start-epoch", type=int, default=0)
    parser.add_argument("--ldam-alpha-max", type=float, default=10.0)
    parser.add_argument(
        "--tail-num-classes",
        type=int,
        default=4,
        help="Number of lowest-support train classes to track for LDAM tail_best.pt.",
    )
    parser.add_argument("--k-folds", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--backbone-backend",
        choices=["auto", "timm", "torchvision"],
        default="auto",
        help="Backbone implementation. auto detects checkpoint backends and defaults to timm when no checkpoint is passed.",
    )
    parser.add_argument(
        "--imagenet-pretrained",
        action="store_true",
        help="Initialize backbones with ImageNet weights before loading any branch checkpoints. Enabled automatically when no branch checkpoints are passed.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["f1_macro", "dice_macro"],
        default="f1_macro",
        help="Validation metric used for best.pt checkpoint selection and LR scheduling.",
    )
    parser.add_argument(
        "--calibrate-bias",
        action="store_true",
        help="Tune per-class logit biases on validation predictions after training and save calibration.json.",
    )
    parser.add_argument(
        "--calibration-metric",
        choices=["f1_macro", "dice_macro"],
        default="dice_macro",
        help="Metric optimized by post-hoc class-bias calibration.",
    )
    parser.add_argument("--calibration-max-bias", type=float, default=1.5)
    parser.add_argument("--calibration-step", type=float, default=0.25)
    parser.add_argument("--calibration-passes", type=int, default=3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--tau", type=float, default=0.0, help="Generalized Balanced Softmax strength in [0, 0.5].")
    parser.add_argument("--lws-epochs", type=int, default=0, help="Number of LWS post-training epochs; 0 disables LWS.")
    parser.add_argument("--lws-lr", type=float, default=1e-2)
    parser.add_argument("--lws-sampler-power", type=float, default=0.5)
    parser.add_argument("--lws-min-scale", type=float, default=0.75)
    parser.add_argument("--lws-max-scale", type=float, default=1.5)
    parser.add_argument("--ema", action="store_true", help="Enable Exponential Moving Average (EMA) for model weights")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Decay rate for EMA")
    parser.add_argument("--fit-temperature", action="store_true", help="Fit one positive validation temperature per checkpoint variant.")
    return parser.parse_args()
