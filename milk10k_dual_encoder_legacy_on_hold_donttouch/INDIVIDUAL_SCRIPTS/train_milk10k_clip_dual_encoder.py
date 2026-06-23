#!/usr/bin/env python3
"""Train MILK10k CLIP-style dual encoder with InfoNCE alignment then fusion fine-tuning."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("clip", "Train MILK10k CLIP-style dual encoder.")
