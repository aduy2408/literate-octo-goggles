#!/usr/bin/env python3
"""Train supervised lightweight MILK10k dual encoder with concat/diff/product fusion."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("fusion_v2", "Train MILK10k supervised fusion dual encoder v2.")
