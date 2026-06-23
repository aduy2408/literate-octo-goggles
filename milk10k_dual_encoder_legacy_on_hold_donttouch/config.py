"""Static architecture and backbone configuration."""

from __future__ import annotations

from dataclasses import dataclass

MONET_COLUMNS = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

MODEL_SPECS: dict[str, tuple[str, int]] = {
    "efficientnet_b0": ("efficientnet_b0", 224),
    "mobilenetv2": ("mobilenetv2_100", 224),
    "convnext_tiny": ("convnext_tiny", 224),
    "resnet50": ("resnet50", 224),
}


@dataclass(frozen=True)
class ArchitectureConfig:
    key: str
    objective: str = "supervised"
    fusion: str = "concat_diff_product"
    shared_encoder: bool = False
    use_metadata: bool = False
    use_attributes: bool = False
    two_stage: bool = False


ARCHITECTURES: dict[str, ArchitectureConfig] = {
    "fusion_v2": ArchitectureConfig("fusion_v2"),
    "multitask": ArchitectureConfig("multitask", objective="multitask"),
    "clip": ArchitectureConfig("clip", objective="clip", two_stage=True),
    "siglip": ArchitectureConfig("siglip", objective="siglip", two_stage=True),
    "sm3": ArchitectureConfig("sm3", objective="sm3", use_attributes=True),
    "siamese": ArchitectureConfig("siamese", shared_encoder=True),
    "late_fusion": ArchitectureConfig("late_fusion", fusion="late_logits"),
    "mil_attention": ArchitectureConfig("mil_attention", fusion="mil_attention"),
    "metadata_fusion": ArchitectureConfig("metadata_fusion", use_metadata=True),
    "cross_attention": ArchitectureConfig("cross_attention", fusion="cross_attention"),
    "partial_channel_attention": ArchitectureConfig("partial_channel_attention", fusion="partial_channel_attention"),
    "partial_cross_attention": ArchitectureConfig("partial_cross_attention", fusion="partial_cross_attention"),
}
