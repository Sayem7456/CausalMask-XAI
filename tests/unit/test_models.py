"""Tests for model factory."""

import pytest
import torch

from causalmask.models.factory import (
    create_model,
    list_available_models,
    get_weight_id,
    BACKBONE_CHOICES,
)


def test_list_models():
    models = list_available_models()
    assert "efficientnet_b0" in models
    assert "resnet18" in models


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        create_model("unknown_model")


def test_efficientnet_b0_output_shape():
    model = create_model("efficientnet_b0", num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2)


def test_resnet18_output_shape():
    model = create_model("resnet18", num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2)


def test_model_weight_id_pretrained():
    model = create_model("efficientnet_b0", num_classes=2, pretrained=True)
    wid = get_weight_id(model)
    assert "IMAGENET1K_V1" in wid


def test_model_weight_id_random():
    model = create_model("resnet18", num_classes=2, pretrained=False)
    wid = get_weight_id(model)
    assert "random" in wid


def test_model_backbone_name():
    model = create_model("efficientnet_b0", num_classes=2, pretrained=False)
    assert model._backbone_name == "efficientnet_b0"
