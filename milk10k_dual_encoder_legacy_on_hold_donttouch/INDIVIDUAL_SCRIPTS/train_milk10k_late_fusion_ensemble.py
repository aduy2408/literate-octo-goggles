#!/usr/bin/env python3
"""Train MILK10k late-fusion dual branch ensemble."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("late_fusion", "Train MILK10k late-fusion ensemble.")
