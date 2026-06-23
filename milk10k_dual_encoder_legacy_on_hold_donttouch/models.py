"""Model components for lightweight MILK10k dual encoders."""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

from milk10k_dual_encoder.config import ArchitectureConfig, MONET_COLUMNS


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=1)


class LightweightDualEncoderModel(nn.Module):
    def __init__(
        self,
        timm_name: str,
        num_classes: int,
        arch: ArchitectureConfig,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        metadata_input_dim: int = 0,
        metadata_dim: int = 64,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.clinical_encoder = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.dermoscopic_encoder = (
            self.clinical_encoder
            if arch.shared_encoder
            else timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        )
        feature_dim = int(self.clinical_encoder.num_features)
        self.clinical_projection = ProjectionHead(feature_dim, projection_dim)
        self.dermoscopic_projection = ProjectionHead(feature_dim, projection_dim)

        self.metadata_encoder: nn.Module | None = None
        metadata_out_dim = 0
        if arch.use_metadata:
            self.metadata_encoder = nn.Sequential(
                nn.LayerNorm(metadata_input_dim),
                nn.Linear(metadata_input_dim, metadata_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            metadata_out_dim = metadata_dim

        pair_dim = feature_dim * 4
        if arch.fusion == "late_logits":
            self.clinical_classifier = nn.Linear(feature_dim, num_classes)
            self.dermoscopic_classifier = nn.Linear(feature_dim, num_classes)
            self.classifier = None
        elif arch.fusion == "mil_attention":
            self.attention = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
            self.classifier = self._classifier(feature_dim + metadata_out_dim, hidden_dim, num_classes, dropout)
        elif arch.fusion == "cross_attention":
            heads = 4 if feature_dim % 4 == 0 else 1
            self.cross_attention = nn.MultiheadAttention(feature_dim, heads, dropout=dropout, batch_first=True)
            self.classifier = self._classifier(feature_dim * 2 + metadata_out_dim, hidden_dim, num_classes, dropout)
        elif arch.fusion == "partial_channel_attention":
            attended_dim = max(1, pair_dim // 2)
            self.partial_channel_dim = attended_dim
            self.channel_attention = nn.Sequential(
                nn.LayerNorm(attended_dim),
                nn.Linear(attended_dim, max(16, attended_dim // 4)),
                nn.GELU(),
                nn.Linear(max(16, attended_dim // 4), attended_dim),
                nn.Sigmoid(),
            )
            self.classifier = self._classifier(pair_dim + metadata_out_dim, hidden_dim, num_classes, dropout)
        elif arch.fusion == "partial_cross_attention":
            reduced_dim = min(hidden_dim, feature_dim)
            heads = 4 if reduced_dim % 4 == 0 else 1
            self.partial_clinical_proj = nn.Linear(feature_dim, reduced_dim)
            self.partial_dermoscopic_proj = nn.Linear(feature_dim, reduced_dim)
            self.partial_cross_attention = nn.MultiheadAttention(reduced_dim, heads, dropout=dropout, batch_first=True)
            self.partial_cross_norm = nn.LayerNorm(reduced_dim * 2)
            self.classifier = self._classifier(pair_dim + reduced_dim * 2 + metadata_out_dim, hidden_dim, num_classes, dropout)
        else:
            self.classifier = self._classifier(pair_dim + metadata_out_dim, hidden_dim, num_classes, dropout)

        self.attribute_head = nn.Linear(feature_dim * 4, len(MONET_COLUMNS)) if arch.use_attributes else None
        
        # Learnable Weight Scaling (LWS) parameter
        self.class_scales = nn.Parameter(torch.ones(num_classes))

    @staticmethod
    def _classifier(in_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clinical_features = self.clinical_encoder(images[:, 0])
        dermoscopic_features = self.dermoscopic_encoder(images[:, 1])
        return clinical_features, dermoscopic_features

    def projections(self, clinical_features: torch.Tensor, dermoscopic_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.clinical_projection(clinical_features), self.dermoscopic_projection(dermoscopic_features)

    @staticmethod
    def pair_features(clinical_features: torch.Tensor, dermoscopic_features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                clinical_features,
                dermoscopic_features,
                torch.abs(clinical_features - dermoscopic_features),
                clinical_features * dermoscopic_features,
            ],
            dim=1,
        )

    def forward(self, images: torch.Tensor, metadata: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        clinical_features, dermoscopic_features = self.encode(images)
        clinical_z, dermoscopic_z = self.projections(clinical_features, dermoscopic_features)
        pair_features = self.pair_features(clinical_features, dermoscopic_features)
        metadata_features = self._metadata_features(metadata)

        if self.arch.fusion == "late_logits":
            logits = (self.clinical_classifier(clinical_features) + self.dermoscopic_classifier(dermoscopic_features)) / 2.0
        elif self.arch.fusion == "mil_attention":
            logits = self._mil_logits(clinical_features, dermoscopic_features, metadata_features)
        elif self.arch.fusion == "cross_attention":
            logits = self._cross_attention_logits(clinical_features, dermoscopic_features, metadata_features)
        elif self.arch.fusion == "partial_channel_attention":
            logits = self._partial_channel_logits(pair_features, metadata_features)
        elif self.arch.fusion == "partial_cross_attention":
            logits = self._partial_cross_attention_logits(
                clinical_features,
                dermoscopic_features,
                pair_features,
                metadata_features,
            )
        else:
            fused = pair_features if metadata_features is None else torch.cat([pair_features, metadata_features], dim=1)
            logits = self.classifier(fused)

        # Apply Learnable Weight Scaling (LWS)
        logits = logits * self.class_scales

        output = {
            "logits": logits,
            "clinical_z": clinical_z,
            "dermoscopic_z": dermoscopic_z,
            "clinical_features": clinical_features,
            "dermoscopic_features": dermoscopic_features,
        }
        if self.attribute_head is not None:
            output["attribute_logits"] = self.attribute_head(pair_features)
        return output

    def _metadata_features(self, metadata: torch.Tensor | None) -> torch.Tensor | None:
        if self.metadata_encoder is None or metadata is None:
            return None
        return self.metadata_encoder(metadata)

    def _mil_logits(self, clinical_features, dermoscopic_features, metadata_features):
        stacked = torch.stack([clinical_features, dermoscopic_features], dim=1)
        weights = torch.softmax(self.attention(stacked), dim=1)
        fused = (stacked * weights).sum(dim=1)
        if metadata_features is not None:
            fused = torch.cat([fused, metadata_features], dim=1)
        return self.classifier(fused)

    def _cross_attention_logits(self, clinical_features, dermoscopic_features, metadata_features):
        stacked = torch.stack([clinical_features, dermoscopic_features], dim=1)
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        fused = attended.reshape(attended.size(0), -1)
        if metadata_features is not None:
            fused = torch.cat([fused, metadata_features], dim=1)
        return self.classifier(fused)

    def _partial_channel_logits(self, pair_features, metadata_features):
        attended = pair_features[:, : self.partial_channel_dim]
        residual = pair_features[:, self.partial_channel_dim :]
        attended = attended * self.channel_attention(attended)
        fused = torch.cat([attended, residual], dim=1)
        if metadata_features is not None:
            fused = torch.cat([fused, metadata_features], dim=1)
        return self.classifier(fused)

    def _partial_cross_attention_logits(self, clinical_features, dermoscopic_features, pair_features, metadata_features):
        clinical = self.partial_clinical_proj(clinical_features)
        dermoscopic = self.partial_dermoscopic_proj(dermoscopic_features)
        stacked = torch.stack([clinical, dermoscopic], dim=1)
        attended, _ = self.partial_cross_attention(stacked, stacked, stacked)
        attended = self.partial_cross_norm(attended.reshape(attended.size(0), -1))
        fused = torch.cat([pair_features, attended], dim=1)
        if metadata_features is not None:
            fused = torch.cat([fused, metadata_features], dim=1)
        return self.classifier(fused)


def set_encoder_trainable(model: LightweightDualEncoderModel, trainable: bool) -> None:
    for param in model.clinical_encoder.parameters():
        param.requires_grad = trainable
    for param in model.dermoscopic_encoder.parameters():
        param.requires_grad = trainable
