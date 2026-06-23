#!/usr/bin/env python3
"""Train SM3-inspired MILK10k dual encoder with contrastive and MONET auxiliary losses."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("sm3", "Train MILK10k SM3-inspired dual encoder.")
