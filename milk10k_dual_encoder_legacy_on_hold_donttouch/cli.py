"""CLI helpers for architecture-specific MILK10k entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from milk10k_dual_encoder.config import ARCHITECTURES, MODEL_SPECS
from milk10k_dual_encoder.training import train_architecture


def parse_common_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("dual_encoder_architecture_runs"))
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--attribute-weight", type=float, default=0.2)
    parser.add_argument("--metadata-dim", type=int, default=64)
    parser.add_argument(
        "--paired-augmentation-manifest",
        type=Path,
        default=None,
        help="Optional paired diffusion augmentation manifest. Rows are appended to train split only.",
    )
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ema", action="store_true", help="Enable Exponential Moving Average")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--tau", type=float, default=0.0, help="Logit adjustment tau for Generalized Balanced Softmax")
    parser.add_argument("--lws-epochs", type=int, default=0, help="Number of epochs for Learnable Weight Scaling post-training")
    return parser.parse_args()


def main_for_architecture(architecture: str, description: str) -> None:
    args = parse_common_args(description)
    args.architecture = architecture
    train_architecture(args)
