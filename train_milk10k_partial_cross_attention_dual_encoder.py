#!/usr/bin/env python3
"""Train MILK10k dual encoder with reduced partial cross-attention fusion."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("partial_cross_attention", "Train MILK10k partial cross-attention dual encoder.")
