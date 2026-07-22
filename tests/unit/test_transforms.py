"""Tests for paired image-mask transforms."""

import torch

from causalmask.data.transforms import (
    PairedCompose,
    paired_horizontal_flip,
    paired_random_rotation,
    to_tensor_image,
    to_tensor_mask,
)


def _make_dummy_image_tensor(batch=False):
    c, h, w = (3, 64, 64)
    t = torch.rand(c, h, w)
    return t


def _make_dummy_mask_tensor():
    h, w = (64, 64)
    t = torch.zeros(1, h, w)
    t[:, 10:30, 20:40] = 1.0
    return t


def test_paired_horizontal_flip_shape():
    img = _make_dummy_image_tensor()
    mask = _make_dummy_mask_tensor()
    transform = paired_horizontal_flip(p=1.0)
    img_out, mask_out = transform(img, mask)
    assert img_out.shape == img.shape
    assert mask_out.shape == mask.shape


def test_paired_horizontal_flip_no_mask():
    img = _make_dummy_image_tensor()
    transform = paired_horizontal_flip(p=1.0)
    img_out, mask_out = transform(img, None)
    assert img_out.shape == img.shape
    assert mask_out is None


def test_paired_random_rotation_shape():
    img = _make_dummy_image_tensor()
    mask = _make_dummy_mask_tensor()
    transform = paired_random_rotation(degrees=10)
    img_out, mask_out = transform(img, mask)
    assert img_out.shape == img.shape
    assert mask_out.shape == mask.shape


def test_paired_compose():
    img = _make_dummy_image_tensor()
    mask = _make_dummy_mask_tensor()
    compose = PairedCompose(
        image_transforms=[paired_horizontal_flip(p=1.0), paired_random_rotation(degrees=5)],
    )
    img_out, mask_out = compose(img, mask)
    assert img_out.shape == img.shape
    assert mask_out.shape == mask.shape


def test_deterministic_eval_transforms():
    from PIL import Image
    import numpy as np
    from causalmask.data.transforms import build_eval_transforms
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    mask = _make_dummy_mask_tensor()
    img_t, paired = build_eval_transforms()
    img_out = img_t(img)
    assert img_out.shape == (3, 64, 64)
    assert torch.isfinite(img_out).all()
    # Paired eval should be identity
    img_p, mask_p = paired(img_out, mask)
    assert torch.equal(img_p, img_out)
    assert torch.equal(mask_p, mask)


def test_to_tensor_image():
    from PIL import Image
    import numpy as np
    pil_img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    t = to_tensor_image(pil_img)
    assert t.shape == (3, 64, 64)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_to_tensor_mask():
    from PIL import Image
    import numpy as np
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    arr[10:30, 20:40] = 255
    pil_mask = Image.fromarray(arr)
    t = to_tensor_mask(pil_mask)
    assert t.shape == (1, 64, 64)
    assert t.dtype == torch.float32
    assert set(t.unique().tolist()).issubset({0.0, 1.0})


def test_train_transforms_accepts_pil_image():
    """Regression test: build_train_transforms must accept PIL Image input.

    Broken by T.ConvertImageDtype which expects Tensor.
    The fix replaces ConvertImageDtype with T.ToTensor.
    """
    from PIL import Image
    import numpy as np
    from causalmask.data.transforms import build_train_transforms

    pil_img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))
    img_t, paired = build_train_transforms()

    result = img_t(pil_img)
    assert isinstance(result, torch.Tensor), "train transform must return Tensor"
    assert result.shape == (3, 100, 100), f"expected (3,100,100) got {result.shape}"
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_eval_transforms_accepts_pil_image():
    """Regression test: build_eval_transforms must accept PIL Image input."""
    from PIL import Image
    import numpy as np
    from causalmask.data.transforms import build_eval_transforms

    pil_img = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype(np.uint8))
    img_t, paired = build_eval_transforms()

    result = img_t(pil_img)
    assert isinstance(result, torch.Tensor), "eval transform must return Tensor"
    assert result.shape == (3, 100, 100), f"expected (3,100,100) got {result.shape}"
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
