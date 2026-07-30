"""RISE (Randomized Input Sampling for Explanation) attribution.

Petsiuk et al. BMVC 2018:
  RISE generates N random binary masks, applies them to the input,
  and computes a weighted average of masks weighted by the model's
  score for the masked input.

Records: n_masks, grid_size, Bernoulli probability, interpolation, seed.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn


logger = logging.getLogger(__name__)

_EPS = 1e-8


class RISE:
    """RISE attribution.

    Generates random binary masks at a lower resolution, upsamples them,
    and produces a weighted average. Supports min-max normalization to
    produce non-negative maps in [0, 1].

    Memory-conscious: processes masks in chunks if needed.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        n_masks: int = 4000,
        grid_size: int = 8,
        bernoulli_prob: float = 0.5,
        interpolation: str = "bilinear",
        mask_chunk_size: int = 500,
        seed: int = 42,
    ):
        if n_masks < 1:
            raise ValueError(f"n_masks must be >= 1, got {n_masks}")
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        if not (0 < bernoulli_prob <= 1.0):
            raise ValueError(
                f"bernoulli_prob must be in (0, 1], got {bernoulli_prob}"
            )
        if interpolation not in ("bilinear", "nearest"):
            raise ValueError(
                f"interpolation must be 'bilinear' or 'nearest', got '{interpolation}'"
            )

        self._model = model
        self._model.eval()
        self._device = device
        self._n_masks = n_masks
        self._grid_size = grid_size
        self._bernoulli_prob = bernoulli_prob
        self._interpolation = interpolation
        self._mask_chunk_size = mask_chunk_size
        self._seed = seed
        self._mask_generator = torch.Generator(device="cpu")

    def _generate_masks(self, input_h: int, input_w: int) -> Tensor:
        self._mask_generator.manual_seed(self._seed)
        gs_h, gs_w = self._grid_size, self._grid_size
        n = self._n_masks

        low_res = torch.rand(
            n, 1, gs_h, gs_w,
            generator=self._mask_generator,
            dtype=torch.float32,
        )
        low_res = (low_res < self._bernoulli_prob).float()

        full_masks = torch.nn.functional.interpolate(
            low_res,
            size=(input_h, input_w),
            mode=self._interpolation,
            align_corners=False if self._interpolation == "bilinear" else None,
        )
        return full_masks

    def attribute(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute RISE attributions.

        Args:
            images: [B, C, H, W].
            target_classes: Optional [B] target class indices.

        Returns:
            Attribution tensor [B, 1, H, W] (non-negative, raw).
        """
        b, c, h, w = images.shape
        images = images.to(self._device)

        masks = self._generate_masks(h, w)
        n_masks = len(masks)

        attributions = torch.zeros(b, 1, h, w, device=self._device)

        with torch.no_grad():
            for start in range(0, n_masks, self._mask_chunk_size):
                end = min(start + self._mask_chunk_size, n_masks)
                chunk_masks = masks[start:end].to(self._device)
                chunk_size = chunk_masks.shape[0]

                masked_images = []
                for i in range(b):
                    img = images[i : i + 1]
                    masked = img * chunk_masks
                    masked_images.append(masked)
                masked_batch = torch.cat(masked_images, dim=0)

                logits = self._model(masked_batch)
                probs = torch.softmax(logits, dim=1)

                if target_classes is None:
                    tc_expanded = logits.argmax(dim=1)
                else:
                    tc = target_classes.to(self._device)
                    tc_expanded = tc.repeat_interleave(chunk_size)

                scores = probs[torch.arange(len(probs)), tc_expanded]

                for i in range(b):
                    i_start = i * chunk_size
                    i_end = (i + 1) * chunk_size
                    sample_scores = scores[i_start:i_end]
                    sample_masks = chunk_masks
                    weighted = (sample_masks * sample_scores.view(-1, 1, 1, 1)).sum(dim=0, keepdim=True)
                    attributions[i : i + 1] += weighted

        avg = attributions / n_masks
        avg = _safe_tensor(avg)

        return avg

    def attribute_batch(
        self,
        images: Tensor,
        target_classes: Optional[Tensor] = None,
    ) -> Tensor:
        return self.attribute(images, target_classes)

    def cleanup(self) -> None:
        pass


def _safe_tensor(t: Tensor) -> Tensor:
    t = t.detach()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def build_rise(
    model: nn.Module,
    device: Optional[torch.device] = None,
    n_masks: int = 4000,
    grid_size: int = 8,
    bernoulli_prob: float = 0.5,
    interpolation: str = "bilinear",
    mask_chunk_size: int = 500,
    seed: int = 42,
) -> RISE:
    """Build a RISE attributor.

    Args:
        model: Classifier model.
        device: torch device.
        n_masks: Number of random masks.
        grid_size: Low-res mask grid size.
        bernoulli_prob: Probability of 1 in each mask cell.
        interpolation: 'bilinear' or 'nearest'.
        mask_chunk_size: Chunk size for memory-conscious processing.
        seed: RNG seed for reproducible masks.

    Returns:
        RISE instance.
    """
    if device is None:
        device = next(model.parameters()).device
    return RISE(
        model, device,
        n_masks=n_masks,
        grid_size=grid_size,
        bernoulli_prob=bernoulli_prob,
        interpolation=interpolation,
        mask_chunk_size=mask_chunk_size,
        seed=seed,
    )
