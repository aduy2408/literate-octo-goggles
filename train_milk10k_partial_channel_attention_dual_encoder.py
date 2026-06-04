#!/usr/bin/env python3
"""Train MILK10k dual encoder with partial channel attention over fused features."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("partial_channel_attention", "Train MILK10k partial channel attention dual encoder.")
