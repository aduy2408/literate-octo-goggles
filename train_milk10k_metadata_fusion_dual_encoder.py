#!/usr/bin/env python3
"""Train MILK10k dual encoder with image and tabular metadata fusion."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("metadata_fusion", "Train MILK10k metadata fusion dual encoder.")
