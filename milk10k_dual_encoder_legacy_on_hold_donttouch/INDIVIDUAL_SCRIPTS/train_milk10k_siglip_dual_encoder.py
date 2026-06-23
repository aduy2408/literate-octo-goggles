#!/usr/bin/env python3
"""Train MILK10k SigLIP-style dual encoder with sigmoid pair alignment then fine-tuning."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("siglip", "Train MILK10k SigLIP-style dual encoder.")
