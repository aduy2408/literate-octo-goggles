import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from data_related.ham10k.train_ham10k_backbone import (
    attach_image_paths,
    build_encoder,
    index_images,
    main,
    split_by_lesion,
    validate_metadata,
)


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def make_dataset(root: Path) -> Path:
    rows = []
    for class_index, label in enumerate(CLASSES):
        for sample_index in range(2):
            image_id = f"image_{class_index}_{sample_index}"
            lesion_id = f"lesion_{class_index}_{sample_index}"
            Image.new("RGB", (24, 24), (class_index * 20, sample_index * 40, 80)).save(
                root / f"{image_id}.jpg"
            )
            rows.append({"image_id": image_id, "lesion_id": lesion_id, "dx": label})
    metadata_path = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path


def write_custom_factory(root: Path) -> Path:
    module_path = root / "tiny_backbone.py"
    module_path.write_text(
        """
import torch.nn as nn

def build_backbone(pretrained=False):
    del pretrained
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ), 4
""".strip(),
        encoding="utf-8",
    )
    return module_path


def test_metadata_image_resolution_and_group_split(tmp_path):
    metadata_path = make_dataset(tmp_path)
    frame = pd.read_csv(metadata_path)
    validate_metadata(frame)
    resolved = attach_image_paths(frame, index_images(tmp_path))
    train, val = split_by_lesion(resolved, val_ratio=0.5, seed=7)

    assert not set(train["lesion_id"]).intersection(val["lesion_id"])
    assert set(train["dx"]) == set(CLASSES)
    assert set(val["dx"]) == set(CLASSES)


def test_missing_columns_and_duplicate_images_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_metadata(pd.DataFrame({"image_id": ["x"]}))

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "a" / "same.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "b" / "same.png")
    with pytest.raises(ValueError, match="Duplicate image IDs"):
        index_images(tmp_path)


@pytest.mark.parametrize("backend,backbone", [("torchvision", "resnet18"), ("timm", "resnet18")])
def test_standard_backbone_forward(backend, backbone):
    if backend == "timm":
        pytest.importorskip("timm")
    args = argparse.Namespace(
        backend=backend,
        backbone=backbone,
        pretrained=False,
        custom_module=None,
        custom_factory="build_backbone",
    )
    encoder, feature_dim = build_encoder(args)
    encoder.eval()
    with torch.no_grad():
        features = encoder(torch.zeros(2, 3, 64, 64))
    assert features.shape == (2, feature_dim)


def test_custom_training_backbone_export_and_resume(tmp_path):
    metadata_path = make_dataset(tmp_path)
    factory_path = write_custom_factory(tmp_path)
    output_dir = tmp_path / "runs"
    common_args = [
        "--data-root", str(tmp_path),
        "--metadata-file", metadata_path.name,
        "--backend", "custom",
        "--backbone", "tiny",
        "--custom-module", str(factory_path),
        "--no-pretrained",
        "--img-size", "16",
        "--batch-size", "7",
        "--num-workers", "0",
        "--val-ratio", "0.5",
        "--output-dir", str(output_dir),
        "--device", "cpu",
        "--no-amp",
    ]
    main(common_args + ["--epochs", "1"])

    run_dir = output_dir / "custom_tiny"
    backbone_checkpoint = torch.load(
        run_dir / "best_backbone.pth", map_location="cpu", weights_only=False
    )
    assert backbone_checkpoint["feature_dim"] == 4
    assert "0.weight" in backbone_checkpoint["encoder_state_dict"]
    assert not any(key.startswith("model.") for key in backbone_checkpoint["encoder_state_dict"])

    main(common_args + ["--epochs", "2", "--resume", str(run_dir / "last_full.pth")])
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    assert [record["epoch"] for record in history] == [1, 2]
