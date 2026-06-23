"""Single-image classifier and optional metadata fusion."""

from __future__ import annotations

import timm
import torch
from torch import nn

METADATA_MODES = ("none", "concat", "gated_concat", "gated_only")


def build_feature_encoder(backbone: str, backend: str, pretrained: bool):
    if backend == "timm":
        encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        return encoder, int(encoder.num_features)
    if backend != "torchvision":
        raise ValueError(f"Unsupported backbone backend: {backend}")
    from torchvision import models
    builders = {
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights),
        "resnet50": (models.resnet50, models.ResNet50_Weights),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights),
    }
    if backbone not in builders:
        raise ValueError(f"torchvision backend does not support backbone={backbone!r}")
    builder, weights_enum = builders[backbone]
    encoder = builder(weights=weights_enum.DEFAULT if pretrained else None)
    if backbone.startswith("efficientnet") or backbone.startswith("convnext"):
        feature_dim = int(encoder.classifier[-1].in_features)
        encoder.classifier = nn.Identity()
    else:
        feature_dim = int(encoder.fc.in_features)
        encoder.fc = nn.Identity()
    return encoder, feature_dim


class MetadataHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, max(32, output_dim * 2)), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(max(32, output_dim * 2), output_dim), nn.GELU(), nn.LayerNorm(output_dim)
        )

    def forward(self, value):
        return self.net(value)


class DermoscopicMetadataClassifier(nn.Module):
    def __init__(
        self, num_classes: int, metadata_input_dim: int, metadata_mode: str = "none", backbone: str = "efficientnet_b2",
        imagenet_pretrained: bool = True, branch_dim: int = 512, metadata_dim: int = 64,
        classifier_hidden_dim: int = 512, dropout: float = 0.3, backbone_backend: str = "timm",
        metadata_gate_hidden_dim: int | None = None,
    ):
        super().__init__()
        if metadata_mode not in METADATA_MODES:
            raise ValueError(f"Unsupported metadata mode: {metadata_mode}")
        self.metadata_mode = metadata_mode
        self.backbone_name = backbone
        self.backbone_backend = backbone_backend
        self.encoder, feature_dim = build_feature_encoder(backbone, backbone_backend, imagenet_pretrained)
        self.image_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, branch_dim), nn.GELU(), nn.LayerNorm(branch_dim))
        if metadata_mode == "none":
            self.metadata_head = None
            self.metadata_gate = None
            classifier_input = branch_dim
        else:
            self.metadata_head = MetadataHead(metadata_input_dim, metadata_dim, dropout)
            if metadata_mode in ("gated_concat", "gated_only"):
                gate_hidden = metadata_gate_hidden_dim or metadata_dim
                self.metadata_gate = nn.Sequential(
                    nn.LayerNorm(metadata_input_dim), nn.Linear(metadata_input_dim, gate_hidden), nn.GELU(),
                    nn.Linear(gate_hidden, branch_dim), nn.Sigmoid()
                )
                nn.init.zeros_(self.metadata_gate[-2].weight)
                nn.init.constant_(self.metadata_gate[-2].bias, 2.0)
            else:
                self.metadata_gate = None
            classifier_input = branch_dim if metadata_mode == "gated_only" else branch_dim + metadata_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input), nn.Dropout(dropout), nn.Linear(classifier_input, classifier_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        features = self.image_head(self.encoder(image))
        if self.metadata_mode == "none":
            fused = features
        else:
            if metadata is None:
                raise ValueError(f"metadata is required for metadata_mode={self.metadata_mode}")
            if self.metadata_gate is not None:
                features = features * self.metadata_gate(metadata)
            fused = features if self.metadata_mode == "gated_only" else torch.cat([features, self.metadata_head(metadata)], dim=1)
        return self.classifier(fused)


def set_encoder_trainable(model: DermoscopicMetadataClassifier, trainable: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = trainable


def set_metadata_trainable(model: DermoscopicMetadataClassifier, trainable: bool) -> None:
    for module in (model.metadata_head, model.metadata_gate):
        if module is not None:
            for parameter in module.parameters(): parameter.requires_grad = trainable
