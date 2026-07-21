"""Model factory for creating classifier backbones.

Supported architectures:
- EfficientNet-B0 (primary backbone)
- ResNet-18 (architecture-generalization baseline)

Both output logits for binary classification.
"""

from __future__ import annotations

import logging
from typing import Any

from torch import Tensor, nn

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}

BACKBONE_CHOICES = ("efficientnet_b0", "resnet18")


def register_model(name: str):
    """Decorator to register a model builder."""
    def _decorator(fn):
        _MODEL_REGISTRY[name] = {"builder": fn}
        return fn
    return _decorator


@register_model("efficientnet_b0")
def build_efficientnet_b0(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Build EfficientNet-B0 classifier.

    Weight identifier (pretrained=True):
        torchvision://efficientnet_b0/EfficientNet_B0_Weights.IMAGENET1K_V1
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    weight_id = (
        "EfficientNet_B0_Weights.IMAGENET1K_V1"
        if pretrained
        else "none (random init)"
    )

    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, num_classes),
    )
    model._weight_id = weight_id  # type: ignore[attr-defined]
    return model


@register_model("resnet18")
def build_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Build ResNet-18 classifier.

    Weight identifier (pretrained=True):
        torchvision://resnet18/ResNet18_Weights.IMAGENET1K_V1
    """
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    weight_id = (
        "ResNet18_Weights.IMAGENET1K_V1" if pretrained else "none (random init)"
    )

    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model._weight_id = weight_id  # type: ignore[attr-defined]
    return model


def create_model(
    backbone: str,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a classifier model by name.

    Args:
        backbone: Model name. Must be in BACKBONE_CHOICES.
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained ImageNet weights.

    Returns:
        PyTorch module.

    Raises:
        ValueError: If backbone is not in BACKBONE_CHOICES.
    """
    if backbone not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    builder = _MODEL_REGISTRY[backbone]["builder"]
    model = builder(num_classes=num_classes, pretrained=pretrained, **kwargs)
    model._backbone_name = backbone  # type: ignore[attr-defined]
    logger.info(
        f"Created model: {backbone}, "
        f"pretrained={pretrained}, "
        f"weight_id={getattr(model, '_weight_id', 'unknown')}, "
        f"num_classes={num_classes}"
    )
    return model


def get_weight_id(model: nn.Module) -> str:
    """Get the weight identifier string from a model if available."""
    return getattr(model, "_weight_id", "unknown")


def list_available_models() -> list[str]:
    """List all registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


def get_model_state(model: nn.Module) -> dict[str, Tensor]:
    """Get model state dict, stripping DDP wrappers if present."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def load_model_weights(model: nn.Module, state_dict: dict[str, Tensor], strict: bool = True) -> None:
    """Load state dict into model, handling DDP prefix."""
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)
