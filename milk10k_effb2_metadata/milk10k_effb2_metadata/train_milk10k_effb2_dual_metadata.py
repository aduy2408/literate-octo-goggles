#!/usr/bin/env python3
"""Train a MILK10k dual EfficientNet-B2 classifier with metadata fusion."""

from milk10k_effb2_metadata.cli import parse_args


def main() -> None:
    args = parse_args()
    from milk10k_effb2_metadata.training import run

    run(args)


if __name__ == "__main__":
    main()
