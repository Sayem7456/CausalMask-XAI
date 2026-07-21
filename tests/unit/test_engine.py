"""Tests for training engine."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from causalmask.training.engine import Trainer, TrainingConfig
from causalmask.training.checkpointing import Checkpoint, save_checkpoint, load_checkpoint


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 2),
        )

    def forward(self, x):
        return self.net(x)


def _make_dummy_loader(batch_size=4, num_samples=16, seed=42):
    torch.manual_seed(seed)
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 2, (num_samples,))

    class _DictDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.images = images
            self.labels = labels
            self.sample_ids = [f"dummy_{i:04d}" for i in range(num_samples)]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return {
                "image": self.images[idx],
                "label": self.labels[idx],
                "sample_id": self.sample_ids[idx],
            }

    return DataLoader(_DictDataset(), batch_size=batch_size)


def test_trainer_initialization():
    model = _SimpleModel()
    config = TrainingConfig(num_epochs=1, amp_enabled=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        trainer = Trainer(model, config, torch.device("cpu"), run_dir)
        assert trainer.epoch == 0
        assert trainer.global_step == 0
        assert trainer.run_dir == run_dir
        assert trainer.checkpoint_dir == run_dir / "checkpoints"


def test_train_one_epoch():
    model = _SimpleModel()
    config = TrainingConfig(
        num_epochs=1,
        early_stopping_patience=50,
        amp_enabled=False,
        gradient_clip_val=10.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        trainer = Trainer(model, config, torch.device("cpu"), run_dir)
        train_loader = _make_dummy_loader(num_samples=16)
        val_loader = _make_dummy_loader(num_samples=8, seed=99)
        result = trainer.fit(train_loader, val_loader)
        assert result["best_model_saved"] is True
        assert result["total_epochs"] >= 1


def test_finite_loss():
    model = _SimpleModel()
    config = TrainingConfig(num_epochs=1, amp_enabled=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        trainer = Trainer(model, config, torch.device("cpu"), run_dir)
        train_loader = _make_dummy_loader(num_samples=8)
        metrics = trainer.train_epoch(train_loader)
        assert torch.isfinite(torch.tensor(metrics["loss"])).item()
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_checkpoint_resume():
    model = _SimpleModel()
    config = TrainingConfig(num_epochs=5, early_stopping_patience=10, amp_enabled=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        trainer = Trainer(model, config, torch.device("cpu"), run_dir)
        train_loader = _make_dummy_loader(num_samples=16)
        val_loader = _make_dummy_loader(num_samples=8, seed=99)

        # Save a checkpoint as if we were resuming from epoch 2
        ckpt_path = run_dir / "resume.pt"
        trainer.model.load_state_dict(model.state_dict())  # ensure model has state
        ckpt = Checkpoint(
            epoch=2,
            global_step=20,
            model_state=trainer.model.state_dict(),
            optimizer_state=trainer.optimizer.state_dict(),
            best_metric=0.6,
            best_epoch=2,
            metrics_history=trainer.history,
        )
        save_checkpoint(ckpt, ckpt_path)
        assert ckpt_path.exists()
        trainer2 = Trainer(_SimpleModel(), config, torch.device("cpu"), run_dir)
        result = trainer2.fit(train_loader, val_loader, resume_path=ckpt_path)
        assert result["total_epochs"] >= 2


def test_prediction_export():
    model = _SimpleModel()
    config = TrainingConfig(num_epochs=1, amp_enabled=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        trainer = Trainer(model, config, torch.device("cpu"), run_dir)
        loader = _make_dummy_loader(batch_size=4, num_samples=8)
        preds = trainer.predict(loader)
        assert len(preds) == 8
        required_cols = ["sample_id", "label", "prob_benign", "prob_malignant", "predicted_class"]
        for col in required_cols:
            assert col in preds.columns, f"Missing column: {col}"
        path = trainer.save_predictions(preds, partition="test")
        assert path.exists()
        import pandas as pd
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == 8
        assert "partition" in reloaded.columns
        assert reloaded["partition"].unique().tolist() == ["test"]
