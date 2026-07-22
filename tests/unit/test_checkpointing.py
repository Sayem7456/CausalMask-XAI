"""Tests for checkpoint save and resume."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from causalmask.training.checkpointing import (
    Checkpoint,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    capture_rng_states,
    restore_rng_states,
)


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def _make_model_and_optim():
    model = _SimpleModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optim


def test_save_and_load_checkpoint():
    model, optim = _make_model_and_optim()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        ckpt = Checkpoint(
            epoch=5,
            global_step=100,
            model_state=model.state_dict(),
            optimizer_state=optim.state_dict(),
            best_metric=0.5,
            best_epoch=5,
        )
        save_checkpoint(ckpt, ckpt_path)
        assert ckpt_path.exists()

        model2, optim2 = _make_model_and_optim()
        loaded = load_checkpoint(ckpt_path, model2, optim2, device=torch.device("cpu"))
        assert loaded.epoch == 5
        assert loaded.global_step == 100
        assert loaded.best_metric == 0.5

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


def test_checkpoint_resume_epoch():
    model, optim = _make_model_and_optim()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir()
        paths = []
        for epoch in [0, 5, 10]:
            p = ckpt_dir / f"epoch_{epoch:04d}.pt"
            ckpt = Checkpoint(epoch=epoch, global_step=epoch * 20, model_state=model.state_dict())
            save_checkpoint(ckpt, p)
            paths.append(p)

        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None
        assert latest.name == "epoch_0010.pt"


def test_find_latest_no_checkpoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        latest = find_latest_checkpoint(Path(tmpdir))
        assert latest is None


def test_rng_capture_restore():
    import random
    states = capture_rng_states()
    rng_b = random.randint(0, 1000)
    restore_rng_states(states)
    rng_c = random.randint(0, 1000)
    assert rng_b == rng_c


def test_save_and_load_with_rng_states():
    """Regression: checkpoint with real rng_states (numpy arrays) must load.

    PyTorch 2.6+ ``torch.load(weights_only=True)`` rejects pickle globals
    from ``numpy._core.multiarray._reconstruct``.  This test verifies both
    the load-side fix (``weights_only=False``) and the save-side fix (numpy
    state stored as bytes).
    """
    import random

    model, optim = _make_model_and_optim()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_with_rng.pt"

        ckpt = Checkpoint(
            epoch=3,
            global_step=50,
            model_state=model.state_dict(),
            optimizer_state=optim.state_dict(),
            rng_states=capture_rng_states(),
        )
        save_checkpoint(ckpt, ckpt_path)
        assert ckpt_path.exists()

        rng_b = random.randint(0, 1000)
        model2, optim2 = _make_model_and_optim()
        loaded = load_checkpoint(
            ckpt_path, model2, optim2, device=torch.device("cpu")
        )
        assert loaded.epoch == 3
        assert loaded.global_step == 50

        restore_rng_states(loaded.rng_states)
        rng_c = random.randint(0, 1000)
        assert rng_b == rng_c, "RNG state not restored correctly"
