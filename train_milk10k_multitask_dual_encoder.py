#!/usr/bin/env python3
"""Train MILK10k dual encoder with classification plus contrastive alignment."""

from milk10k_dual_encoder_common import main_for_architecture


if __name__ == "__main__":
    main_for_architecture("multitask", "Train MILK10k multitask dual encoder.")
