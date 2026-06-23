"""Model components for the dual EfficientNet-B2 metadata classifier."""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

ONE_ENCODER_IMAGE_FUSIONS = ("single_encoder_canvas", "shared_encoder_pool")


def is_one_encoder_image_fusion(image_fusion: str) -> bool:
    return image_fusion in ONE_ENCODER_IMAGE_FUSIONS


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BranchClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedExpertClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetadataHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(out_dim * 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata)


class MetadataChannelGate(nn.Module):
    def __init__(self, metadata_input_dim: int, channel_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(metadata_input_dim)
        self.fc1 = nn.Linear(metadata_input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, channel_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 2.0)

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        gate = self.norm(metadata)
        gate = self.fc1(gate)
        gate = self.act(gate)
        gate = self.dropout(gate)
        gate = torch.sigmoid(self.fc2(gate))
        return gate


class DualEffB2MetadataClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        metadata_input_dim: int,
        branch_dim: int,
        metadata_dim: int,
        classifier_hidden_dim: int,
        dropout: float,
        imagenet_pretrained: bool,
        clinical_backbone_backend: str,
        dermoscopic_backbone_backend: str,
        backbone: str = "efficientnet_b2",
        disable_metadata: bool = False,
        metadata_fusion: str = "concat",
        image_fusion: str = "concat",
        metadata_gate_hidden_dim: int | None = None,
        classifier_style: str = "legacy",
        logit_fusion_mode: str = "single",
        fusion_logit_weight: float = 0.6,
        clinical_logit_weight: float = 0.2,
        dermoscopic_logit_weight: float = 0.2,
    ) -> None:
        super().__init__()
        if metadata_fusion not in ("concat", "gated_concat", "gated_only"):
            raise ValueError(f"Unsupported metadata_fusion: {metadata_fusion}")
        if image_fusion not in (
            "concat",
            "cross_attention",
            "co_attention",
            "compact_bilinear",
            "low_rank_bilinear",
            "adaptive_gate",
            "moe",
            "shared_private",
            *ONE_ENCODER_IMAGE_FUSIONS,
        ):
            raise ValueError(f"Unsupported image_fusion: {image_fusion}")
        if logit_fusion_mode not in ("single", "fixed"):
            raise ValueError(f"Unsupported logit_fusion_mode: {logit_fusion_mode}")
        if is_one_encoder_image_fusion(image_fusion) and logit_fusion_mode != "single":
            raise ValueError(f"{image_fusion} supports only --logit-fusion-mode single.")
        if classifier_style not in ("legacy", "simple"):
            raise ValueError(f"Unsupported classifier_style: {classifier_style}")
        self.clinical_backbone_backend = clinical_backbone_backend
        self.dermoscopic_backbone_backend = dermoscopic_backbone_backend
        self.backbone = normalize_backbone_name(backbone)
        self.disable_metadata = disable_metadata
        self.metadata_dim = metadata_dim
        self.metadata_fusion = metadata_fusion
        self.image_fusion = image_fusion
        self.classifier_style = classifier_style
        self.logit_fusion_mode = logit_fusion_mode
        self.fusion_logit_weight = fusion_logit_weight
        self.clinical_logit_weight = clinical_logit_weight
        self.dermoscopic_logit_weight = dermoscopic_logit_weight
        self.one_encoder = is_one_encoder_image_fusion(image_fusion)
        if self.one_encoder:
            if clinical_backbone_backend != dermoscopic_backbone_backend:
                raise ValueError(f"{image_fusion} requires one shared backend, got different branch backends.")
            self.shared_encoder, shared_feature_dim = build_feature_encoder(
                backbone,
                clinical_backbone_backend,
                imagenet_pretrained,
            )
            clinical_feature_dim = shared_feature_dim
            dermoscopic_feature_dim = shared_feature_dim
        else:
            self.clinical_encoder, clinical_feature_dim = build_feature_encoder(
                backbone,
                clinical_backbone_backend,
                imagenet_pretrained,
            )
            self.dermoscopic_encoder, dermoscopic_feature_dim = build_feature_encoder(
                backbone,
                dermoscopic_backbone_backend,
                imagenet_pretrained,
            )

        self.clinical_head = ProjectionHead(clinical_feature_dim, branch_dim, dropout)
        self.dermoscopic_head = ProjectionHead(dermoscopic_feature_dim, branch_dim, dropout)
        if self.one_encoder:
            self.shared_head = ProjectionHead(shared_feature_dim, branch_dim, dropout)
        self.metadata_head = MetadataHead(metadata_input_dim, metadata_dim, dropout)
        if metadata_fusion in ("gated_concat", "gated_only"):
            gate_hidden_dim = metadata_gate_hidden_dim if metadata_gate_hidden_dim is not None else metadata_dim
            if self.one_encoder:
                self.shared_metadata_gate = MetadataChannelGate(
                    metadata_input_dim,
                    shared_feature_dim,
                    gate_hidden_dim,
                    dropout,
                )
            else:
                self.clinical_metadata_gate = MetadataChannelGate(
                    metadata_input_dim,
                    clinical_feature_dim,
                    gate_hidden_dim,
                    dropout,
                )
                self.dermoscopic_metadata_gate = MetadataChannelGate(
                    metadata_input_dim,
                    dermoscopic_feature_dim,
                    gate_hidden_dim,
                    dropout,
                )
        metadata_output_dim = 0 if metadata_fusion == "gated_only" else metadata_dim
        fused_dim = self._fusion_dim(branch_dim, metadata_output_dim, image_fusion)
        heads = 4 if branch_dim % 4 == 0 else 1
        if image_fusion == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(branch_dim, heads, dropout=dropout, batch_first=True)
            self.cross_attention_norm = nn.LayerNorm(branch_dim * 2)
        elif image_fusion == "co_attention":
            self.co_attention = nn.MultiheadAttention(branch_dim, heads, dropout=dropout, batch_first=True)
            self.co_attention_norm = nn.LayerNorm(branch_dim * 4)
        elif image_fusion in ("compact_bilinear", "low_rank_bilinear"):
            self.bilinear_clinical = nn.Linear(branch_dim, branch_dim)
            self.bilinear_dermoscopic = nn.Linear(branch_dim, branch_dim)
            self.bilinear_norm = nn.LayerNorm(branch_dim)
        elif image_fusion == "adaptive_gate":
            gate_input_dim = branch_dim * 2 + metadata_output_dim
            self.image_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, max(branch_dim, 64)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(branch_dim, 64), branch_dim),
                nn.Sigmoid(),
            )
        elif image_fusion == "moe":
            clinical_expert_dim = branch_dim + metadata_output_dim
            dermoscopic_expert_dim = branch_dim + metadata_output_dim
            joint_expert_dim = branch_dim * 2 + metadata_output_dim
            router_dim = branch_dim * 2 + metadata_output_dim
            self.clinical_expert = GatedExpertClassifier(clinical_expert_dim, classifier_hidden_dim, num_classes, dropout)
            self.dermoscopic_expert = GatedExpertClassifier(
                dermoscopic_expert_dim,
                classifier_hidden_dim,
                num_classes,
                dropout,
            )
            self.joint_expert = GatedExpertClassifier(joint_expert_dim, classifier_hidden_dim, num_classes, dropout)
            self.expert_router = nn.Sequential(
                nn.LayerNorm(router_dim),
                nn.Dropout(dropout),
                nn.Linear(router_dim, 3),
            )
        elif image_fusion == "shared_private":
            if clinical_feature_dim != dermoscopic_feature_dim:
                raise ValueError("shared_private image fusion requires matching branch feature dimensions.")
            self.shared_head = ProjectionHead(clinical_feature_dim, branch_dim, dropout)
        self.classifier = (
            None
            if image_fusion == "moe"
            else self._classifier(fused_dim, classifier_hidden_dim, num_classes, dropout, classifier_style)
        )
        if logit_fusion_mode == "fixed":
            self.clinical_classifier = BranchClassifier(branch_dim, num_classes, dropout)
            self.dermoscopic_classifier = BranchClassifier(branch_dim, num_classes, dropout)
        else:
            self.clinical_classifier = None
            self.dermoscopic_classifier = None
            
        # LWS is a post-training stage. Keep scales frozen during normal
        # representation/classifier training and enable them explicitly later.
        self.class_scales = nn.Parameter(torch.ones(num_classes), requires_grad=False)

    @staticmethod
    def _classifier(
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        classifier_style: str,
    ) -> nn.Sequential:
        if classifier_style == "simple":
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _fusion_dim(branch_dim: int, metadata_dim: int, image_fusion: str) -> int:
        if image_fusion in ("concat", "cross_attention"):
            image_dim = branch_dim * 2
        elif image_fusion in (
            "compact_bilinear",
            "low_rank_bilinear",
            "adaptive_gate",
            "shared_private",
            "shared_encoder_pool",
        ):
            image_dim = branch_dim * 3
        elif image_fusion == "single_encoder_canvas":
            image_dim = branch_dim
        elif image_fusion == "co_attention":
            image_dim = branch_dim * 4
        elif image_fusion == "moe":
            image_dim = branch_dim * 2
        else:
            raise ValueError(f"Unsupported image_fusion: {image_fusion}")
        return image_dim + metadata_dim

    def forward(
        self,
        clinical: torch.Tensor,
        dermoscopic: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        if self.one_encoder:
            fusion_logits = self._one_encoder_logits(clinical, dermoscopic, metadata)
            return fusion_logits * self.class_scales

        if self.metadata_fusion in ("gated_concat", "gated_only"):
            clinical_features = self.encode_with_metadata_gate(
                self.clinical_encoder,
                self.clinical_backbone_backend,
                clinical,
                metadata,
                self.clinical_metadata_gate,
            )
            dermoscopic_features = self.encode_with_metadata_gate(
                self.dermoscopic_encoder,
                self.dermoscopic_backbone_backend,
                dermoscopic,
                metadata,
                self.dermoscopic_metadata_gate,
            )
        else:
            clinical_features = self.clinical_encoder(clinical)
            dermoscopic_features = self.dermoscopic_encoder(dermoscopic)
        clinical_features = torch.flatten(clinical_features, 1)
        dermoscopic_features = torch.flatten(dermoscopic_features, 1)
        clinical_repr = self.clinical_head(clinical_features)
        dermoscopic_repr = self.dermoscopic_head(dermoscopic_features)
        metadata_repr = None
        if self.metadata_fusion == "gated_only":
            metadata_repr = None
        else:
            if self.disable_metadata:
                metadata_repr = clinical_repr.new_zeros((clinical_repr.size(0), self.metadata_dim))
            else:
                metadata_repr = self.metadata_head(metadata)
        if self.image_fusion == "moe":
            fusion_logits = self._moe_logits(clinical_repr, dermoscopic_repr, metadata_repr)
        else:
            fused = self._fused_features(clinical_features, dermoscopic_features, clinical_repr, dermoscopic_repr, metadata_repr)
            fusion_logits = self.classifier(fused)
        if self.logit_fusion_mode != "fixed":
            return fusion_logits * self.class_scales
        clinical_logits = self.clinical_classifier(clinical_repr)
        dermoscopic_logits = self.dermoscopic_classifier(dermoscopic_repr)
        return (
            self.fusion_logit_weight * fusion_logits
            + self.clinical_logit_weight * clinical_logits
            + self.dermoscopic_logit_weight * dermoscopic_logits
        ) * self.class_scales

    def _one_encoder_logits(
        self,
        clinical: torch.Tensor,
        dermoscopic: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        if self.image_fusion == "single_encoder_canvas":
            combined = torch.cat([clinical, dermoscopic], dim=-1)
            features = self._encode_shared(combined, metadata)
            repr_ = self.shared_head(features)
            fused_image = repr_
        elif self.image_fusion == "shared_encoder_pool":
            clinical_features = self._encode_shared(clinical, metadata)
            dermoscopic_features = self._encode_shared(dermoscopic, metadata)
            clinical_repr = self.shared_head(clinical_features)
            dermoscopic_repr = self.shared_head(dermoscopic_features)
            fused_image = torch.cat(
                [
                    0.5 * (clinical_repr + dermoscopic_repr),
                    torch.abs(clinical_repr - dermoscopic_repr),
                    clinical_repr * dermoscopic_repr,
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported one-encoder image_fusion: {self.image_fusion}")

        metadata_repr = self._metadata_repr(fused_image, metadata)
        return self.classifier(self._append_metadata(fused_image, metadata_repr))

    def _encode_shared(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if self.metadata_fusion in ("gated_concat", "gated_only"):
            features = self.encode_with_metadata_gate(
                self.shared_encoder,
                self.clinical_backbone_backend,
                images,
                metadata,
                self.shared_metadata_gate,
            )
        else:
            features = self.shared_encoder(images)
        return torch.flatten(features, 1)

    def _metadata_repr(self, image_repr: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor | None:
        if self.metadata_fusion == "gated_only":
            return None
        if self.disable_metadata:
            return image_repr.new_zeros((image_repr.size(0), self.metadata_dim))
        return self.metadata_head(metadata)

    def _append_metadata(self, features: torch.Tensor, metadata_repr: torch.Tensor | None) -> torch.Tensor:
        if metadata_repr is None:
            return features
        return torch.cat([features, metadata_repr], dim=1)

    def _fused_features(
        self,
        clinical_features: torch.Tensor,
        dermoscopic_features: torch.Tensor,
        clinical_repr: torch.Tensor,
        dermoscopic_repr: torch.Tensor,
        metadata_repr: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.image_fusion == "concat":
            fused = torch.cat([clinical_repr, dermoscopic_repr], dim=1)
        elif self.image_fusion == "cross_attention":
            tokens = torch.stack([clinical_repr, dermoscopic_repr], dim=1)
            attended, _ = self.cross_attention(tokens, tokens, tokens)
            fused = self.cross_attention_norm(attended.reshape(attended.size(0), -1))
        elif self.image_fusion == "co_attention":
            tokens = torch.stack([clinical_repr, dermoscopic_repr], dim=1)
            attended, _ = self.co_attention(tokens, tokens, tokens)
            updated = tokens + attended
            fused = torch.cat(
                [
                    clinical_repr,
                    dermoscopic_repr,
                    updated[:, 0],
                    updated[:, 1],
                ],
                dim=1,
            )
            fused = self.co_attention_norm(fused)
        elif self.image_fusion in ("compact_bilinear", "low_rank_bilinear"):
            bilinear = self.bilinear_clinical(clinical_repr) * self.bilinear_dermoscopic(dermoscopic_repr)
            bilinear = self.bilinear_norm(bilinear)
            fused = torch.cat([clinical_repr, dermoscopic_repr, bilinear], dim=1)
        elif self.image_fusion == "adaptive_gate":
            gate_input = torch.cat([clinical_repr, dermoscopic_repr], dim=1)
            if metadata_repr is not None:
                gate_input = torch.cat([gate_input, metadata_repr], dim=1)
            gate = self.image_gate(gate_input)
            gated = gate * clinical_repr + (1.0 - gate) * dermoscopic_repr
            fused = torch.cat([gated, torch.abs(clinical_repr - dermoscopic_repr), clinical_repr * dermoscopic_repr], dim=1)
        elif self.image_fusion == "shared_private":
            clinical_shared = self.shared_head(clinical_features)
            dermoscopic_shared = self.shared_head(dermoscopic_features)
            shared = 0.5 * (clinical_shared + dermoscopic_shared)
            fused = torch.cat([clinical_repr, dermoscopic_repr, shared], dim=1)
        else:
            raise ValueError(f"Unsupported image_fusion: {self.image_fusion}")
        return self._append_metadata(fused, metadata_repr)

    def _moe_logits(
        self,
        clinical_repr: torch.Tensor,
        dermoscopic_repr: torch.Tensor,
        metadata_repr: torch.Tensor | None,
    ) -> torch.Tensor:
        clinical_input = self._append_metadata(clinical_repr, metadata_repr)
        dermoscopic_input = self._append_metadata(dermoscopic_repr, metadata_repr)
        joint_input = self._append_metadata(torch.cat([clinical_repr, dermoscopic_repr], dim=1), metadata_repr)
        expert_logits = torch.stack(
            [
                self.clinical_expert(clinical_input),
                self.dermoscopic_expert(dermoscopic_input),
                self.joint_expert(joint_input),
            ],
            dim=1,
        )
        router_weights = torch.softmax(self.expert_router(joint_input), dim=1)
        return (expert_logits * router_weights[:, :, None]).sum(dim=1)

    def encode_with_metadata_gate(
        self,
        encoder: nn.Module,
        backbone_backend: str,
        images: torch.Tensor,
        metadata: torch.Tensor,
        gate_module: MetadataChannelGate,
    ) -> torch.Tensor:
        feature_map = extract_spatial_features(encoder, backbone_backend, self.backbone, images)
        if self.disable_metadata:
            gate = feature_map.new_ones((feature_map.size(0), feature_map.size(1)))
        else:
            gate = gate_module(metadata).to(device=feature_map.device, dtype=feature_map.dtype)
        gated = feature_map * gate[:, :, None, None]
        return F.adaptive_avg_pool2d(gated, 1)


class DualConvNeXtMetadataClassifier(DualEffB2MetadataClassifier):
    """Dual-image metadata classifier backed by independent ConvNeXt Base encoders."""

    def __init__(self, *args, **kwargs) -> None:
        backbone = normalize_backbone_name(kwargs.pop("backbone", "convnext_base"))
        if backbone != "convnext_base":
            raise ValueError(
                "DualConvNeXtMetadataClassifier only supports the convnext_base backbone, "
                f"got {backbone!r}."
            )
        super().__init__(*args, backbone=backbone, **kwargs)


def normalize_backbone_name(name: str) -> str:
    name = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if name in ("tfefficientnetv2b2", "efficientnetv2b2", "effnetv2b2", "effv2b2"):
        return "tf_efficientnetv2_b2"
    if name in ("efficientnetb2", "effnetb2", "effb2"):
        return "efficientnet_b2"
    if name in ("efficientnetb1", "effnetb1", "effb1"):
        return "efficientnet_b1"
    if name in ("resnet50", "resnet_50"):
        return "resnet50"
    if name in ("convnextbase", "convxbase"):
        return "convnext_base"
    raise ValueError(f"Unknown backbone: {name}")


def model_class_for_backbone(backbone: str) -> type[DualEffB2MetadataClassifier]:
    """Return the dedicated model class for a normalized backbone name."""
    backbone = normalize_backbone_name(backbone)
    if backbone == "convnext_base":
        return DualConvNeXtMetadataClassifier
    return DualEffB2MetadataClassifier


def default_image_size(backbone: str) -> int:
    """Return the training/inference resolution used when --image-size is omitted."""
    backbone = normalize_backbone_name(backbone)
    if backbone == "efficientnet_b2":
        return 260
    if backbone == "tf_efficientnetv2_b2":
        return 384
    if backbone == "efficientnet_b1":
        return 240
    if backbone == "convnext_base":
        return 384
    return 224


def resolve_image_size(backbone: str, image_size: int | None) -> int:
    """Use an explicit image size when provided, otherwise use the backbone default."""
    return int(image_size) if image_size is not None else default_image_size(backbone)


def extract_spatial_features(encoder: nn.Module, backbone_backend: str, backbone: str, images: torch.Tensor) -> torch.Tensor:
    if backbone_backend == "timm":
        features = encoder.forward_features(images)
        if isinstance(features, (tuple, list)):
            features = features[-1]
    elif backbone_backend == "torchvision":
        if backbone in ("efficientnet_b2", "efficientnet_b1", "convnext_base"):
            features = encoder.features(images)
        elif backbone == "resnet50":
            features = encoder.conv1(images)
            features = encoder.bn1(features)
            features = encoder.relu(features)
            features = encoder.maxpool(features)
            features = encoder.layer1(features)
            features = encoder.layer2(features)
            features = encoder.layer3(features)
            features = encoder.layer4(features)
        else:
            raise ValueError(f"Unsupported torchvision backbone for gated fusion: {backbone}")
    else:
        raise ValueError(f"Unsupported backbone backend: {backbone_backend}")

    if features.ndim != 4:
        raise RuntimeError(
            f"Expected spatial feature map [B, C, H, W] for gated fusion, got shape {tuple(features.shape)}"
        )
    return features


def build_feature_encoder(backbone: str, backbone_backend: str, imagenet_pretrained: bool) -> tuple[nn.Module, int]:
    backbone = normalize_backbone_name(backbone)
    if backbone_backend == "timm":
        model = timm.create_model(
            backbone,
            pretrained=imagenet_pretrained,
            num_classes=0,
            global_pool="avg",
        )
        return model, int(model.num_features)

    if backbone_backend == "torchvision":
        if backbone == "tf_efficientnetv2_b2":
            raise ValueError("tf_efficientnetv2_b2 is only available with --backbone-backend timm.")
        if backbone == "efficientnet_b2":
            from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            model = efficientnet_b2(weights=weights)
            feature_dim = int(model.classifier[1].in_features)
            model.classifier = nn.Identity()
            return model, feature_dim
        elif backbone == "efficientnet_b1":
            from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
            weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            model = efficientnet_b1(weights=weights)
            feature_dim = int(model.classifier[1].in_features)
            model.classifier = nn.Identity()
            return model, feature_dim
        elif backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            model = resnet50(weights=weights)
            feature_dim = int(model.fc.in_features)
            model.fc = nn.Identity()
            return model, feature_dim
        elif backbone == "convnext_base":
            from torchvision.models import convnext_base, ConvNeXt_Base_Weights
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
            model = convnext_base(weights=weights)
            feature_dim = int(model.classifier[2].in_features)
            model.classifier = nn.Identity()
            return model, feature_dim
        else:
            raise ValueError(f"Unsupported torchvision backbone: {backbone}")

    raise ValueError(f"Unsupported backbone backend: {backbone_backend}")


def set_encoder_trainable(model: DualEffB2MetadataClassifier, trainable: bool) -> None:
    if getattr(model, "one_encoder", False):
        for param in model.shared_encoder.parameters():
            param.requires_grad = trainable
        return
    for param in model.clinical_encoder.parameters():
        param.requires_grad = trainable
    for param in model.dermoscopic_encoder.parameters():
        param.requires_grad = trainable
