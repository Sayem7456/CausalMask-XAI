"""Causal trainer extending the baseline Trainer with counterfactual losses.

Handles:
- Sequential forward passes (original, sufficient, removed, swapped)
- Causal loss computation with warm-up / ramp
- Gradient accumulation for memory efficiency
- Extended monitoring (divergence, eligibility, confidence)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from causalmask.training.engine import Trainer, TrainingConfig
from causalmask.training.losses import (
    CausalLossConfig,
    compute_causal_losses,
    divergence_between,
    prediction_confidence,
    prediction_entropy,
)
from causalmask.training.checkpointing import Checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


class CausalTrainer(Trainer):
    """Trainer with causal regularisation losses.

    Extends the standard Trainer to:
    - Accept counterfactual transforms as callables
    - Run sequential forward passes per batch
    - Compute CE + sufficiency + background + necessity loss
    - Log causal-specific monitoring metrics
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        run_dir: Path,
        causal_loss_config: CausalLossConfig,
        counterfactual_fn: Any = None,
        checkpoint_dir: Path | None = None,
    ):
        super().__init__(model, config, device, run_dir, checkpoint_dir)
        self.causal_cfg = causal_loss_config
        self.counterfactual_fn = counterfactual_fn

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        metrics_accum = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "suff_loss": 0.0,
            "bg_loss": 0.0,
            "nec_loss": 0.0,
            "nec_eligible": 0.0,
            "accuracy": 0.0,
        }
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            masks = batch.get("mask")
            if masks is not None:
                masks = masks.to(self.device)

            # Generate counterfactuals on-the-fly or use precomputed
            sufficient_img = None
            removed_img = None
            swapped_img = None

            if self.counterfactual_fn is not None:
                cf_result = self.counterfactual_fn(images, masks, labels)
                if isinstance(cf_result, dict):
                    sufficient_img = cf_result.get("sufficient")
                    removed_img = cf_result.get("removed")
                    swapped_img = cf_result.get("swapped")

            self.optimizer.zero_grad()

            # Sequential forward passes for original + counterfactuals
            with self._amp_context():
                original_logits = self.model(images)

                sufficient_logits = None
                if sufficient_img is not None:
                    sufficient_logits = self.model(sufficient_img)

                removed_logits = None
                if removed_img is not None:
                    removed_logits = self.model(removed_img)

                swapped_logits = None
                if swapped_img is not None:
                    swapped_logits = self.model(swapped_img)

                loss_dict = compute_causal_losses(
                    original_logits=original_logits,
                    sufficient_logits=sufficient_logits,
                    removed_logits=removed_logits,
                    swapped_logits=swapped_logits,
                    labels=labels,
                    config=self.causal_cfg,
                    epoch=self.epoch,
                )
                loss = loss_dict["total_loss"]

            # Backward pass
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
            num_batches += 1

            metrics_accum["loss"] += loss.item()
            metrics_accum["ce_loss"] += loss_dict["ce_loss"].item()
            metrics_accum["suff_loss"] += loss_dict["sufficiency_loss"].item()
            metrics_accum["bg_loss"] += loss_dict["background_loss"].item()
            metrics_accum["nec_loss"] += loss_dict["necessity_loss"].item()
            metrics_accum["nec_eligible"] += loss_dict["necessity_eligible_fraction"].item()

            preds = original_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        return {
            "loss": metrics_accum["loss"] / max(num_batches, 1),
            "ce_loss": metrics_accum["ce_loss"] / max(num_batches, 1),
            "suff_loss": metrics_accum["suff_loss"] / max(num_batches, 1),
            "bg_loss": metrics_accum["bg_loss"] / max(num_batches, 1),
            "nec_loss": metrics_accum["nec_loss"] / max(num_batches, 1),
            "nec_eligible": metrics_accum["nec_eligible"] / max(num_batches, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_ce = 0.0
        total_suff = 0.0
        total_bg = 0.0
        total_nec = 0.0
        total_eligible = 0.0
        num_batches = 0

        # Accumulate for monitoring + AUROC
        suff_divergences: list[float] = []
        swap_divergences: list[float] = []
        removed_confidences: list[float] = []
        all_val_logits: list[Tensor] = []
        all_val_labels: list[Tensor] = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            masks = batch.get("mask")
            if masks is not None:
                masks = masks.to(self.device)

            sufficient_img = None
            removed_img = None
            swapped_img = None

            if self.counterfactual_fn is not None:
                cf_result = self.counterfactual_fn(images, masks, labels)
                if isinstance(cf_result, dict):
                    sufficient_img = cf_result.get("sufficient")
                    removed_img = cf_result.get("removed")
                    swapped_img = cf_result.get("swapped")

            original_logits = self.model(images)

            sufficient_logits = None
            if sufficient_img is not None:
                sufficient_logits = self.model(sufficient_img)
                div = divergence_between(original_logits, sufficient_logits)
                suff_divergences.append(div.item())

            removed_logits = None
            if removed_img is not None:
                removed_logits = self.model(removed_img)
                conf = prediction_confidence(removed_logits).mean()
                removed_confidences.append(conf.item())

            swapped_logits = None
            if swapped_img is not None:
                swapped_logits = self.model(swapped_img)
                div = divergence_between(original_logits, swapped_logits)
                swap_divergences.append(div.item())

            loss_dict = compute_causal_losses(
                original_logits=original_logits,
                sufficient_logits=sufficient_logits,
                removed_logits=removed_logits,
                swapped_logits=swapped_logits,
                labels=labels,
                config=self.causal_cfg,
                epoch=self.epoch,
            )

            total_loss += loss_dict["total_loss"].item()
            total_ce += loss_dict["ce_loss"].item()
            total_suff += loss_dict["sufficiency_loss"].item()
            total_bg += loss_dict["background_loss"].item()
            total_nec += loss_dict["necessity_loss"].item()
            total_eligible += loss_dict["necessity_eligible_fraction"].item()

            preds = original_logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            num_batches += 1

            all_val_logits.append(original_logits.detach().cpu())
            all_val_labels.append(labels.detach().cpu())

        n = max(num_batches, 1)

        metrics = {
            "loss": total_loss / n,
            "ce_loss": total_ce / n,
            "suff_loss": total_suff / n,
            "bg_loss": total_bg / n,
            "nec_loss": total_nec / n,
            "nec_eligible": total_eligible / n,
            "accuracy": total_correct / max(total_samples, 1),
        }

        if suff_divergences:
            metrics["suff_divergence"] = sum(suff_divergences) / len(suff_divergences)
        if swap_divergences:
            metrics["swap_divergence"] = sum(swap_divergences) / len(swap_divergences)
        if removed_confidences:
            metrics["removed_confidence"] = sum(removed_confidences) / len(removed_confidences)

        if all_val_logits:
            val_logits_cat = torch.cat(all_val_logits, dim=0)
            val_labels_cat = torch.cat(all_val_labels, dim=0).to(val_logits_cat.device)
            val_probs = torch.softmax(val_logits_cat, dim=1)
            try:
                from sklearn.metrics import roc_auc_score, balanced_accuracy_score
                val_labels_np = val_labels_cat.numpy()
                val_probs_pos = val_probs[:, 1].numpy()
                if len(np.unique(val_labels_np)) > 1:
                    metrics["val_auroc"] = float(roc_auc_score(val_labels_np, val_probs_pos))
                else:
                    metrics["val_auroc"] = float("nan")
                val_preds_np = (val_probs_pos >= 0.5).astype(np.int64)
                metrics["val_balanced_accuracy"] = float(balanced_accuracy_score(val_labels_np, val_preds_np))
            except Exception:
                metrics["val_auroc"] = float("nan")
                metrics["val_balanced_accuracy"] = float("nan")

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Path | None = None,
    ) -> dict[str, Any]:
        from causalmask.training.checkpointing import load_checkpoint

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
            f"Starting causal training: {self.config.num_epochs} epochs max, "
            f"loss_variant={self.causal_cfg.loss_variant}, "
            f"warmup={self.causal_cfg.necessity_warmup_epochs}, "
            f"ramp={self.causal_cfg.necessity_ramp_epochs}"
        )

        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            epoch_time = time.time() - epoch_start

            # Compute gradient norm for monitoring
            total_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # Prediction entropy on validation
            self.model.eval()
            val_entropies = []
            val_confidences = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    logits = self.model(images)
                    ent = prediction_entropy(logits)
                    conf = prediction_confidence(logits).mean()
                    val_entropies.append(ent.item())
                    val_confidences.append(conf.item())

            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_ce": train_metrics["ce_loss"],
                "train_suff": train_metrics["suff_loss"],
                "train_bg": train_metrics["bg_loss"],
                "train_nec": train_metrics["nec_loss"],
                "train_nec_eligible": train_metrics["nec_eligible"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_auroc": val_metrics.get("val_auroc", float("nan")),
                "val_balanced_accuracy": val_metrics.get("val_balanced_accuracy", float("nan")),
                "val_suff_div": val_metrics.get("suff_divergence", 0.0),
                "val_swap_div": val_metrics.get("swap_divergence", 0.0),
                "val_removed_conf": val_metrics.get("removed_confidence", 0.0),
                "val_entropy": sum(val_entropies) / max(len(val_entropies), 1),
                "val_confidence": sum(val_confidences) / max(len(val_confidences), 1),
                "grad_norm": round(total_grad_norm, 6),
                "time_seconds": round(epoch_time, 2),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(entry)

            log_msg = (
                f"Epoch {epoch}: "
                f"loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"suff_div={val_metrics.get('suff_divergence', 0):.4f} "
                f"nec_elig={train_metrics['nec_eligible']:.3f} "
                f"grad_norm={total_grad_norm:.4f} "
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
                rng_states={},
                metrics_history=self.history,
            )
            save_checkpoint(ckpt, self.checkpoint_dir / f"epoch_{epoch:04d}.pt", is_best=is_best)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                else:
                    self.scheduler.step()

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

        self._save_training_history()
        last_epoch = self.history[-1]["epoch"] if self.history else start_epoch
        result = {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "total_epochs": last_epoch + 1,
            "early_stopped": self.patience_counter >= self.config.early_stopping_patience,
            "best_model_saved": self.best_model_state is not None,
        }
        self.save_result_summary(result)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return result
