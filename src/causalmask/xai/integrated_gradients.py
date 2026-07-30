"""Integrated Gradients attribution method.

Integrated Gradients (Sundararajan et al. ICML 2017):
  IG_i(x) = (x_i - x'_i) * ∫_0^1 ∂F(x' + α(x - x'))/∂x_i dα

Records: baseline type, integration steps, convergence delta.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn


logger = logging.getLogger(__name__)

_EPS = 1e-8


class IntegratedGradientsMethod:
    """Integrated Gradients attribution.

    Supports: zero, gaussian_noise, blurred baselines.
    Approximates the integral via Riemann sum with configurable steps.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        steps: int = 50,
        baseline_type: str = "zero",
        internal_batch_size: int = 1,
    ):
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")
        if baseline_type not in ("zero", "gaussian_noise", "blurred"):
            raise ValueError(
                f"baseline_type must be 'zero', 'gaussian_noise', or 'blurred', "
                f"got '{baseline_type}'"
            )

        self._model = model
        self._model.eval()
        self._device = device
        self._steps = steps
        self._baseline_type = baseline_type
        self._internal_batch_size = internal_batch_size

    def _make_baseline(self, x: Tensor) -> Tensor:
        if self._baseline_type == "zero":
            return torch.zeros_like(x)
        elif self._baseline_type == "gaussian_noise":
            return torch.randn_like(x) * 0.1
        else:
            k = max(1, min(x.shape[2], x.shape[3]) // 16)
            if k % 2 == 0:
                k += 1
            return _gaussian_blur_batch(x, kernel_size=k, sigma=k / 2.0)

    def attribute(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[float]]:
        """Compute Integrated Gradients attributions.

        Args:
            images: [B, C, H, W].
            target_classes: Optional [B] target class indices.

        Returns:
            (attribution, convergence_delta) tuple.
              - attribution: [B, 1, H, W] (channel-aggregated, unsigned).
              - convergence_delta: float or None.
        """
        images = images.to(self._device)
        baseline = self._make_baseline(images)

        if target_classes is None:
            with torch.no_grad():
                logits = self._model(images)
            tc = logits.argmax(dim=1).to(self._device)
        else:
            tc = target_classes.to(self._device)

        attr = _riemann_sum(self._model, images, baseline, tc, self._steps, self._device)
        attr = _safe_tensor(attr)

        convergence_delta = _compute_convergence_delta(
            self._model, images, baseline, tc, self._steps, self._device,
        )

        return attr, convergence_delta

    def attribute_batch(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[float]]:
        return self.attribute(images, target_classes)

    def cleanup(self) -> None:
        pass


def _gaussian_blur_batch(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    from torchvision.transforms.functional import gaussian_blur
    b, c, h, w = x.shape
    flat = x.view(b * c, 1, h, w)
    blurred = gaussian_blur(flat, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return blurred.view(b, c, h, w)


def _safe_tensor(t: Tensor) -> Tensor:
    t = t.detach()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def _compute_convergence_delta(
    model: nn.Module,
    images: Tensor,
    baseline: Tensor,
    target_classes: Tensor,
    steps: int,
    device: torch.device,
) -> Optional[float]:
    """Estimate convergence by comparing with twice the steps (no recursion)."""
    try:
        attr_fine = _riemann_sum(model, images, baseline, target_classes, steps * 2, device)
        attr_coarse = _riemann_sum(model, images, baseline, target_classes, steps, device)
        diff = (attr_fine - attr_coarse).abs().mean().item()
        return float(diff)
    except Exception as e:
        logger.warning(f"Convergence delta computation failed: {e}")
        return None


def _riemann_sum(
    model: nn.Module,
    images: Tensor,
    baseline: Tensor,
    target_classes: Tensor,
    steps: int,
    device: torch.device,
) -> Tensor:
    """Compute IG via Riemann approximation without convergence delta."""
    b, c, h, w = images.shape
    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    total_grad = torch.zeros(b, c, h, w, device=device)
    model.zero_grad()

    for alpha in alphas:
        interpolated = baseline + alpha * (images - baseline)
        interpolated = interpolated.detach().requires_grad_(True)

        logits = model(interpolated)

        one_hot = torch.zeros(b, logits.shape[1], device=device)
        one_hot.scatter_(1, target_classes.unsqueeze(1), 1.0)

        model.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=False)
        grad = interpolated.grad
        if grad is None:
            raise RuntimeError("Input gradient is None")
        total_grad = total_grad + grad.detach()

    avg_grad = total_grad / steps
    integrated = (images - baseline) * avg_grad
    return integrated.abs().sum(dim=1, keepdim=True)


def build_integrated_gradients(
    model: nn.Module,
    device: Optional[torch.device] = None,
    steps: int = 50,
    baseline_type: str = "zero",
    internal_batch_size: int = 1,
) -> IntegratedGradientsMethod:
    """Build an Integrated Gradients attributor.

    Args:
        model: Classifier model.
        device: torch device.
        steps: Riemann sum steps.
        baseline_type: 'zero', 'gaussian_noise', or 'blurred'.
        internal_batch_size: For batching through the model.

    Returns:
        IntegratedGradientsMethod instance.
    """
    if device is None:
        device = next(model.parameters()).device
    method = IntegratedGradientsMethod(
        model, device,
        steps=steps,
        baseline_type=baseline_type,
        internal_batch_size=internal_batch_size,
    )
    return method
