"""Training engine for classification models.

Supports:
- Cross-entropy training
- Optional CUDA automatic mixed precision
- Gradient clipping
- Validation-based early stopping
- Per-epoch metric logging
- Checkpoint save and resume
- Per-sample prediction export
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from causalmask.training.checkpointing import (
    Checkpoint,
    capture_rng_states,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    gradient_clip_val: float = 1.0
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    label_smoothing: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


class Trainer:
    """Standard classification trainer with early stopping and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        run_dir: Path,
        checkpoint_dir: Path | None = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dir = run_dir
        self.checkpoint_dir = checkpoint_dir or run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self.scaler: torch.cuda.amp.GradScaler | None = None
        if config.amp_enabled and device.type == "cuda":
            self.scaler = torch.amp.GradScaler(device="cuda")

        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") if config.early_stopping_mode == "min" else -float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: list[dict[str, float]] = []
        self.best_model_state: dict[str, Tensor] | None = None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self) -> Any:
        if self.config.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            )
        elif self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
            )
        elif self.config.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _amp_context(self):
        if self.device.type == "cuda" and self.scaler is not None:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return torch.amp.autocast(device_type="cpu", enabled=False)

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with self._amp_context():
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )
                self.optimizer.step()

            self.global_step += 1
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_logits: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_sample_ids: list[str] = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_sample_ids.extend(batch.get("sample_id", [f"sample_{i}" for i in range(images.size(0))]))

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Path | None = None,
    ) -> dict[str, Any]:
        """Run training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            resume_path: Optional checkpoint to resume from.

        Returns:
            Dict with training results.
        """
        start_epoch = 0

        if resume_path is not None and resume_path.exists():
            ckpt = load_checkpoint(
                resume_path, self.model, self.optimizer, self.scheduler, self.scaler, self.device
            )
            start_epoch = ckpt.epoch
            self.global_step = ckpt.global_step
            self.best_metric = ckpt.best_metric or self.best_metric
            self.best_epoch = ckpt.best_epoch or 0
            self.history = ckpt.metrics_history or []
            logger.info(f"Resumed from epoch {start_epoch}, step {self.global_step}")

        logger.info(
            f"Starting training: {self.config.num_epochs} epochs max, "
            f"early stopping on {self.config.early_stopping_metric} "
            f"(patience={self.config.early_stopping_patience})"
        )

        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_time = time.time() - epoch_start

            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "time_seconds": round(epoch_time, 2),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(entry)

            log_msg = (
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"lr={entry['learning_rate']:.6f} "
                f"({epoch_time:.1f}s)"
            )

            current_metric = val_metrics["loss"]

            is_best = False
            if self.config.early_stopping_mode == "min":
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    is_best = True
                    log_msg += " *best"
                else:
                    self.patience_counter += 1
            else:
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    is_best = True
                    log_msg += " *best"
                else:
                    self.patience_counter += 1

            logger.info(log_msg)

            ckpt = Checkpoint(
                epoch=epoch,
                global_step=self.global_step,
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
                scaler_state=self.scaler.state_dict() if self.scaler is not None else None,
                best_metric=self.best_metric,
                best_epoch=self.best_epoch,
                rng_states=capture_rng_states(),
                metrics_history=self.history,
            )
            save_checkpoint(
                ckpt,
                self.checkpoint_dir / f"epoch_{epoch:04d}.pt",
                is_best=is_best,
            )

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self.patience_counter} epochs)"
                )
                break

        self._save_training_history()

        result = {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "total_epochs": epoch + 1,
            "early_stopped": self.patience_counter >= self.config.early_stopping_patience,
            "best_model_saved": self.best_model_state is not None,
        }
        self.save_result_summary(result)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return result

    def _save_training_history(self) -> None:
        path = self.run_dir / "history.csv"
        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)
        logger.info(f"Saved training history: {path}")

    def save_result_summary(self, result: dict[str, Any]) -> None:
        path = self.run_dir / "status.json"
        summary = {
            **result,
            "config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "num_epochs": self.config.num_epochs,
                "early_stopping_patience": self.config.early_stopping_patience,
                "gradient_clip_val": self.config.gradient_clip_val,
                "amp_enabled": self.config.amp_enabled and self.device.type == "cuda",
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
            },
            "device": str(self.device),
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved result summary: {path}")

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
    ) -> pd.DataFrame:
        """Generate predictions for all samples in a dataloader.

        Returns:
            DataFrame with columns: sample_id, label, prob_benign, prob_malignant,
            predicted_class, logit_benign, logit_malignant.
        """
        self.model.eval()
        records = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            sample_ids = batch.get("sample_id", [f"sample_{i}" for i in range(images.size(0))])

            logits = self.model(images)
            probabilities = torch.softmax(logits, dim=1)

            for i in range(images.size(0)):
                records.append({
                    "sample_id": sample_ids[i],
                    "label": labels[i].item(),
                    "prob_benign": probabilities[i, 0].item(),
                    "prob_malignant": probabilities[i, 1].item(),
                    "predicted_class": logits[i].argmax().item(),
                    "logit_benign": logits[i, 0].item(),
                    "logit_malignant": logits[i, 1].item(),
                })

        df = pd.DataFrame(records)
        df["run_epoch"] = self.epoch
        return df

    def save_predictions(
        self,
        predictions_df: pd.DataFrame,
        partition: str = "test",
    ) -> Path:
        """Save predictions to parquet.

        Args:
            predictions_df: DataFrame from predict().
            partition: Partition name (e.g., 'test', 'validation').

        Returns:
            Path to saved parquet file.
        """
        predictions_df["partition"] = partition
        path = self.run_dir / f"predictions_{partition}.parquet"
        predictions_df.to_parquet(path, index=False)
        logger.info(f"Saved {partition} predictions: {path} ({len(predictions_df)} samples)")
        return path
