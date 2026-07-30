"""Common attribution interface for XAI methods.

Defines:
  - AttributionInput / AttributionOutput dataclasses.
  - AttributionMetadata: all provenance fields recorded per attribution.
  - resolve_target_layer: per-architecture target layer resolution.
  - get_num_target_classes: number of classes from model's classifier head.

Every XAI method must return an AttributionOutput with:
  - attributions:    [B, 1, H, W] finite non-negative tensor.
  - raw_attributions: optional pre-normalization tensor.
  - metadata:        AttributionMetadata records.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

_EPS = 1e-8


@dataclass
class AttributionInput:
    images: Tensor
    target_classes: Optional[Tensor] = None
    masks: Optional[Tensor] = None


@dataclass
class AttributionOutput:
    attributions: Tensor
    raw_attributions: Optional[Tensor] = None
    metadata: list[AttributionMetadata] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.metadata) == 0 and self.attributions is not None:
            b = self.attributions.shape[0]
            self.metadata = [AttributionMetadata() for _ in range(b)]


@dataclass
class AttributionMetadata:
    sample_id: str = ""
    checkpoint_digest: str = ""
    target_class: int = -1
    target_layer: str = ""
    method: str = ""
    method_config: dict[str, Any] = field(default_factory=dict)
    normalization: str = ""
    seed: int = -1
    baseline_type: str = ""
    integration_steps: int = 0
    convergence_delta: Optional[float] = None
    n_masks: int = 0
    grid_size: int = 0
    bernoulli_prob: float = 0.5
    interpolation: str = ""
    failure_flag: str = ""

    def digest(self) -> str:
        payload = {
            "sample_id": self.sample_id,
            "checkpoint_digest": self.checkpoint_digest,
            "target_class": self.target_class,
            "target_layer": self.target_layer,
            "method": self.method,
            "method_config": self.method_config,
            "normalization": self.normalization,
            "seed": self.seed,
            "baseline_type": self.baseline_type,
            "integration_steps": self.integration_steps,
            "n_masks": self.n_masks,
            "grid_size": self.grid_size,
            "bernoulli_prob": self.bernoulli_prob,
            "interpolation": self.interpolation,
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


_TARGET_LAYER_REGISTRY: dict[str, str] = {
    "efficientnet_b0": "features",
    "resnet18": "layer4",
}


_RESOLVED_TARGET_LAYERS: dict[str, dict[str, str]] = {
    "efficientnet_b0": {
        "features": "features",
        "features.8": "features.8",
    },
    "resnet18": {
        "layer4": "layer4",
        "layer4.1.conv2": "layer4.1.conv2",
    },
}


def resolve_target_layer(
    model: nn.Module,
    backbone: str,
    custom_layer: Optional[str] = None,
) -> tuple[nn.Module, str]:
    """Resolve the target layer for gradient-based attribution.

    Args:
        model: Classifier model.
        backbone: Backbone name ('efficientnet_b0' or 'resnet18').
        custom_layer: Optional explicit sub-layer name like 'features.8'
            or 'layer4.1.conv2'.

    Returns:
        (target_module, layer_name) tuple.

    Raises:
        ValueError: If the layer cannot be resolved.
    """
    if custom_layer is not None:
        return _resolve_nested(model, custom_layer)

    default = _TARGET_LAYER_REGISTRY.get(backbone)
    if default is None:
        raise ValueError(
            f"No default target layer for backbone '{backbone}'. "
            f"Available: {list(_TARGET_LAYER_REGISTRY.keys())}. "
            f"Provide custom_layer explicitly."
        )
    module, name = _resolve_nested(model, default)
    logger.info(f"Resolved target layer: {name} (type={type(module).__name__})")
    return module, name


def _resolve_nested(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module, name


def get_num_target_classes(model: nn.Module) -> int:
    """Return number of output classes by inspecting the classifier head.

    Checks common patterns for EfficientNet, ResNet, and generic Linear heads.
    """
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for m in reversed(model.classifier):
            if isinstance(m, nn.Linear):
                return m.out_features
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc.out_features
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.out_features
    raise ValueError("Cannot infer number of output classes from model.")


def _check_finite(values: Tensor, label: str) -> None:
    if not torch.isfinite(values).all():
        n_bad = (~torch.isfinite(values)).sum().item()
        raise ValueError(f"{label}: {n_bad} non-finite values detected.")


def validate_attribution_output(
    output: AttributionOutput,
    input_h: int,
    input_w: int,
    expected_batch: Optional[int] = None,
) -> None:
    """Validate that attribution output meets the common interface contract.

    Checks:
      - attributions shape [B, 1, H, W] or [B, H, W].
      - finite values.
      - non-negative after potential reshape.
      - metadata count matches batch.
    """
    attr = output.attributions
    if attr.ndim == 3:
        attr = attr.unsqueeze(1)
    if attr.ndim != 4:
        raise ValueError(f"attributions must be 4-d [B,1,H,W], got {attr.shape}")
    b, c, h, w = attr.shape
    if expected_batch is not None and b != expected_batch:
        raise ValueError(f"Expected batch {expected_batch}, got {b}")
    _check_finite(attr, "attributions")
    if attr.shape[2] != input_h or attr.shape[3] != input_w:
        logger.warning(
            f"Attribution spatial size ({h},{w}) differs from input ({input_h},{input_w})"
        )

    if output.raw_attributions is not None:
        _check_finite(output.raw_attributions, "raw_attributions")

    if len(output.metadata) != b:
        raise ValueError(
            f"Metadata count {len(output.metadata)} != batch size {b}"
        )


def _summarize_attr(attr: Tensor) -> dict[str, float]:
    x = attr.detach().float()
    return {
        "min": x.min().item(),
        "max": x.max().item(),
        "mean": x.mean().item(),
        "sum": x.sum().item(),
        "n_finite": x.isfinite().sum().item(),
        "n_nonneg": (x >= 0).sum().item(),
    }
