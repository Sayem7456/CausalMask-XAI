"""Paired image-mask transforms for breast ultrasound data.

Training augmentation must:
- Preserve lesion context (no aggressive cropping)
- Be applied identically to image and mask geometrically
- Use nearest-neighbour interpolation for masks
- Include only mild ultrasound-compatible perturbations

Validation/test transforms must be deterministic.
"""

from __future__ import annotations

from typing import Callable

import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision import transforms as T


class PairedCompose:
    """Apply paired transforms to images and masks together."""

    def __init__(self, image_transforms: list[Callable], mask_transforms: list[Callable] | None = None):
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __call__(self, image: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        for t in self.image_transforms:
            if isinstance(t, PairedTransform):
                image, mask = t(image, mask)
            else:
                image = t(image)
        if mask is not None and self.mask_transforms is not None:
            for t in self.mask_transforms:
                if isinstance(t, PairedTransform):
                    _, mask = t(image, mask)
                else:
                    mask = t(mask)
        return image, mask


class PairedTransform:
    """Base for transforms that must operate identically on image and mask."""

    def __init__(self, transform_fn: Callable[[Tensor, Tensor | None], tuple[Tensor, Tensor | None]]):
        self.transform_fn = transform_fn

    def __call__(self, image: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        return self.transform_fn(image, mask)


def paired_horizontal_flip(p: float = 0.5) -> PairedTransform:
    """Random horizontal flip applied identically to image and mask."""
    def _apply(image: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if torch.rand(1).item() < p:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask
    return PairedTransform(_apply)


def paired_random_rotation(degrees: float = 10) -> PairedTransform:
    """Random small rotation applied identically to image and mask."""
    def _apply(image: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        angle = float(torch.empty(1).uniform_(-degrees, degrees).item())
        image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
        if mask is not None:
            mask = F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST, fill=0)
        return image, mask
    return PairedTransform(_apply)


def paired_random_affine(
    degrees: float = 0,
    translate: tuple[float, float] | None = None,
    scale: tuple[float, float] | None = None,
) -> PairedTransform:
    """Random affine transform applied identically to image and mask."""
    def _apply(image: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        params = T.RandomAffine.get_params(
            degrees=[-degrees, degrees] if degrees > 0 else [],
            translate=translate,
            scale_ranges=scale,
            shears=None,
            img_size=image.shape[-2:],
        )
        image = F.affine(
            image, *params,
            interpolation=F.InterpolationMode.BILINEAR, fill=0,
        )
        if mask is not None:
            mask = F.affine(
                mask, *params,
                interpolation=F.InterpolationMode.NEAREST, fill=0,
            )
        return image, mask
    return PairedTransform(_apply)


def build_train_transforms(
    input_size: tuple[int, int] = (224, 224),
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    hflip_prob: float = 0.5,
    rotation_degrees: float = 10,
    translate_max: float = 0.05,
    scale_min: float = 0.95,
    scale_max: float = 1.05,
    gamma_range: tuple[float, float] = (0.9, 1.1),
    contrast_range: tuple[float, float] = (0.9, 1.1),
    noise_std: float = 0.005,
) -> tuple[Callable, Callable]:
    """Build train-time paired augmentations.

    Returns (image_only_transform, paired_transform) where
    image_only_transform handles normalization and noise,
    and paired_transform handles geometric augmentations.
    """
    image_only = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=mean, std=std),
    ])

    paired_transforms = [
        paired_horizontal_flip(p=hflip_prob),
        paired_random_rotation(degrees=rotation_degrees),
    ]
    if rotation_degrees > 0 or translate_max > 0 or scale_min != 1.0 or scale_max != 1.0:
        paired_transforms.append(
            paired_random_affine(
                degrees=rotation_degrees,
                translate=(translate_max, translate_max),
                scale=(scale_min, scale_max),
            ),
        )
    
    paired = PairedCompose(image_transforms=paired_transforms)
    return image_only, paired


def build_eval_transforms(
    input_size: tuple[int, int] = (224, 224),
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> tuple[Callable, Callable]:
    """Build deterministic validation/test transforms.

    Returns (image_only_transform, paired_transform).
    Paired transform is a no-op for evaluation.
    """
    image_only = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=mean, std=std),
    ])

    paired = PairedCompose(image_transforms=[], mask_transforms=[])
    return image_only, paired


def resize_with_padding(
    image: Tensor,
    mask: Tensor | None,
    size: tuple[int, int],
) -> tuple[Tensor, Tensor | None]:
    """Resize image and mask maintaining aspect ratio with zero padding."""
    h, w = image.shape[-2:]
    target_h, target_w = size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    image = F.resize(image, [new_h, new_w], interpolation=F.InterpolationMode.BILINEAR)
    if mask is not None:
        mask = F.resize(mask, [new_h, new_w], interpolation=F.InterpolationMode.NEAREST)

    padding_h = target_h - new_h
    padding_w = target_w - new_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left

    image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    if mask is not None:
        mask = F.pad(mask, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

    return image, mask


def to_tensor_image(pil_image) -> Tensor:
    """Convert PIL Image (RGB) to float32 tensor [C, H, W] in [0, 1]."""
    return F.pil_to_tensor(pil_image).float() / 255.0


def to_tensor_mask(pil_mask) -> Tensor:
    """Convert PIL mask (L) to float32 tensor [1, H, W] in {0, 1}."""
    t = F.pil_to_tensor(pil_mask).float()
    t = (t > 127).float()
    return t
