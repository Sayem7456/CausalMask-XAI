"""Grad-CAM and Grad-CAM++ attribution methods.

Both methods follow the common interface:
  attribute(model, images, target_classes, device, config) → AttributionOutput

Grad-CAM:     Selvaraju et al. ICCV 2017
Grad-CAM++:   Chattopadhyay et al. WACV 2018

Key difference: Grad-CAM uses mean-pooled gradients as weights;
Grad-CAM++ uses higher-order gradient-weighted activations that
capture multiple occurrences of the target class within an image.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import Tensor, nn

from causalmask.xai.base import (
    resolve_target_layer,
)

logger = logging.getLogger(__name__)

_EPS = 1e-8


class _GradCAMBase(nn.Module):
    """Shared forward/backward hook logic for Grad-CAM and Grad-CAM++."""

    def __init__(self, model: nn.Module, target_layer: nn.Module, detach_gradients: bool = True, detach_activations: bool = True):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[Tensor] = None
        self._gradients: Optional[Tensor] = None
        self._detach_gradients = detach_gradients
        self._detach_activations = detach_activations
        self._hook_handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _forward_hook(_m: nn.Module, _inp: Any, out: Tensor) -> None:
            self._activations = out.detach() if self._detach_activations else out

        def _backward_hook(_m: nn.Module, _ginp: Any, gout: Any) -> None:
            g = gout[0] if isinstance(gout, tuple) else gout
            if self._detach_gradients:
                g = g.detach()
            self._gradients = g

        self._hook_handles.append(self.target_layer.register_forward_hook(_forward_hook))
        self._hook_handles.append(self.target_layer.register_full_backward_hook(_backward_hook))

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _get_activations(self) -> Tensor:
        if self._activations is None:
            raise RuntimeError("No activations captured. Run forward pass first.")
        return self._activations

    def _get_gradients(self) -> Tensor:
        if self._gradients is None:
            raise RuntimeError("No gradients captured. Run backward pass first.")
        return self._gradients


class GradCAM:
    """Grad-CAM attribution.

    Computes:
      w_c^k = (1/Z) * sum_i sum_j (dy^c / dA_ij^k)          [mean-pooled gradients]
      L^c = ReLU( sum_k w_c^k * A^k )                         [weighted combination]
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ):
        self._gradcam_module = _GradCAMBase(model, target_layer)
        self._gradcam_module.to(device)
        self._device = device

    def attribute(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Grad-CAM attributions.

        Args:
            images: [B, C, H, W] input tensor.
            target_classes: Optional [B] tensor of target class indices.
                If None, uses predicted class.

        Returns:
            Raw attribution tensor [B, 1, H_out, W_out] at target-layer
            spatial resolution (NOT yet resized to input resolution).
        """
        gm = self._gradcam_module
        gm.zero_grad()
        gm.model.zero_grad()

        images = images.to(self._device).requires_grad_(True)
        logits = gm(images)

        if target_classes is None:
            target_classes = logits.argmax(dim=1)
        else:
            target_classes = target_classes.to(self._device)

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_classes.unsqueeze(1), 1.0)
        logits.backward(gradient=one_hot, retain_graph=False)

        activations = gm._get_activations()
        gradients = gm._get_gradients()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        return cam.detach()

    def attribute_batch(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> Tensor:
        """Alias for attribute()."""
        return self.attribute(images, target_classes)

    def cleanup(self) -> None:
        self._gradcam_module.remove_hooks()


class GradCAMPlusPlus:
    """Grad-CAM++ attribution.

    Computes higher-order gradients by manually propagating through
    the model twice to keep the computation graph intact.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ):
        self._model = model
        self._model.eval()
        self._target_layer = target_layer
        self._device = device
        self._activations: Optional[Tensor] = None
        self._hook_handle = target_layer.register_forward_hook(self._capture_forward)

    def _capture_forward(self, _m: nn.Module, _inp: Any, out: Tensor) -> None:
        self._activations = out

    def cleanup(self) -> None:
        self._hook_handle.remove()

    def attribute(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> Tensor:
        results = []
        for i in range(images.shape[0]):
            single = images[i : i + 1]
            tc = (
                target_classes[i : i + 1]
                if target_classes is not None
                else None
            )
            cam = self._attribute_single(single, tc)
            results.append(cam)
        return torch.cat(results, dim=0)

    def _attribute_single(
        self,
        image: Tensor,
        target_class: Optional[Tensor] = None,
    ) -> Tensor:
        self._model.zero_grad()
        self._activations = None

        x = image.to(self._device).requires_grad_(True)
        logits = self._model(x)

        activations = self._activations
        if activations is None:
            raise RuntimeError("No activations captured.")

        if target_class is None:
            target_class = logits.argmax(dim=1)
        score = logits[0, target_class[0]]

        first_grads = torch.autograd.grad(
            outputs=score,
            inputs=activations,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        first_grads_sq = first_grads.pow(2).sum()

        second_grads = torch.autograd.grad(
            outputs=first_grads_sq,
            inputs=activations,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]

        if second_grads is None:
            second_grads = torch.zeros_like(activations)

        third_grads_out = torch.autograd.grad(
            outputs=second_grads.pow(2).sum(),
            inputs=activations,
            retain_graph=False,
            create_graph=False,
            only_inputs=True,
            allow_unused=True,
        )

        third_grads = third_grads_out[0] if third_grads_out[0] is not None else torch.zeros_like(activations)

        eps_val = 1e-6
        first_grads_clamped = first_grads.clamp(min=0).detach()

        sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        two_second = 2.0 * second_grads.pow(2)
        sum_acts_third = sum_activations * third_grads
        denom = two_second + sum_acts_third + eps_val

        alpha = second_grads.pow(2) / denom

        weights = (alpha * first_grads_clamped).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations.detach()).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = _safe_tensor(cam)
        return cam

    def attribute_batch(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> Tensor:
        return self.attribute(images, target_classes)


def _safe_tensor(t: Tensor) -> Tensor:
    t = t.detach()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def build_gradcam(
    model: nn.Module,
    backbone: str = "efficientnet_b0",
    custom_layer: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> GradCAM:
    """Build a GradCAM attributor.

    Args:
        model: Classifier model.
        backbone: Backbone name.
        custom_layer: Optional explicit sub-layer.
        device: torch device.

    Returns:
        GradCAM instance.
    """
    if device is None:
        device = next(model.parameters()).device
    target_layer, layer_name = resolve_target_layer(model, backbone, custom_layer)
    return GradCAM(model, target_layer, device)


def build_gradcam_plusplus(
    model: nn.Module,
    backbone: str = "efficientnet_b0",
    custom_layer: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> GradCAMPlusPlus:
    """Build a GradCAM++ attributor."""
    if device is None:
        device = next(model.parameters()).device
    target_layer, layer_name = resolve_target_layer(model, backbone, custom_layer)
    return GradCAMPlusPlus(model, target_layer, device)
