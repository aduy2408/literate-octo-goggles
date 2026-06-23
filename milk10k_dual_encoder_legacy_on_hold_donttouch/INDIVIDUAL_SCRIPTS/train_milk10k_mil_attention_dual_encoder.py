#!/usr/bin/env python3
"""Train MILK10k two-view MIL attention dual encoder."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("mil_attention", "Train MILK10k MIL attention dual encoder.")
