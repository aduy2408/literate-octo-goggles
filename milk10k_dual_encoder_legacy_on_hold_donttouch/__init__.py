"""Lightweight MILK10k dual-encoder training package."""

from milk10k_dual_encoder.cli import main_for_architecture
from milk10k_dual_encoder.config import ARCHITECTURES, MODEL_SPECS, MONET_COLUMNS, ArchitectureConfig
from milk10k_dual_encoder.training import train_architecture

__all__ = [
    "ARCHITECTURES",
    "MODEL_SPECS",
    "MONET_COLUMNS",
    "ArchitectureConfig",
    "main_for_architecture",
    "train_architecture",
]
