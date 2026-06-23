#!/usr/bin/env python3
"""Train shared-encoder MILK10k paired-image control model."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("siamese", "Train MILK10k shared Siamese encoder control.")
