"""Causal training losses for CausalMask-XAI.

Implements the four loss terms from the proposal:

    L = L_CE + lambda_s * L_sufficiency + lambda_b * L_background + lambda_n * L_necessity

Each causal loss uses a detached original-image prediction as teacher
to prevent trivial co-adaptation.  All functions operate on logits or
probabilities so they can be tested independently of a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CausalLossConfig:
    """Configuration for all causal training losses.

    Attributes:
        ce_weight: Weight for cross-entropy base loss (=1.0 by default).
        sufficiency_weight: lambda_s.  Weight for sufficiency consistency.
        background_weight: lambda_b.  Weight for background consistency.
        necessity_weight: lambda_n.  Target weight for necessity ranking.
        necessity_margin: m in max(0, p_y(x_removed) - p_y(x) + m).
        necessity_warmup_epochs: Freeze necessity loss for first N epochs.
        necessity_confidence_threshold: Only apply necessity loss when
            original predicted probability for true class >= this value.
        necessity_ramp_epochs: Number of epochs over which to linearly
            ramp the necessity weight from 0 to lambda_n after warm-up.
        use_detached_teacher: If True (default), stop gradients on the
            original prediction used as teacher in sufficiency/background.
        kl_epsilon: Small constant for numerical stability in KL.
        label_smoothing: Label smoothing for cross-entropy.
        loss_variant: One of 'ce_only', 'necessity_only', 'sufficiency_only',
            'background_only', 'full'.  Useful for ablations.
    """
    ce_weight: float = 1.0
    sufficiency_weight: float = 0.5
    background_weight: float = 0.5
    necessity_weight: float = 0.5
    necessity_margin: float = 0.1
    necessity_warmup_epochs: int = 5
    necessity_confidence_threshold: float = 0.6
    necessity_ramp_epochs: int = 5
    use_detached_teacher: bool = True
    kl_epsilon: float = 1e-8
    label_smoothing: float = 0.0
    loss_variant: str = "full"


# ---------------------------------------------------------------------------
# Core loss functions
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Standard cross-entropy classification loss.

    Args:
        logits: [B, C] model logits.
        labels: [B] integer class indices.
        label_smoothing: [0, 1) smoothing factor.

    Returns:
        Scalar loss averaged over batch.
    """
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def _kl_divergence(
    teacher_probs: Tensor,
    student_logits: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """KL divergence D_KL(teacher_probs || softmax(student_logits)).

    teacher_probs is used as the *first* argument of D_KL, meaning we
    compute Σ teacher * log(teacher / student).  Gradients flow only
    through student_logits (unless the caller already detached teacher_probs).

    Args:
        teacher_probs: [B, C] probability distribution (may be detached).
        student_logits: [B, C] model logits for the counterfactual image.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar KL averaged over batch.
    """
    student_log_probs = F.log_softmax(student_logits, dim=1)
    student_probs = student_log_probs.exp()
    # Clamp for numerical safety
    teacher_probs = teacher_probs.clamp(min=epsilon)
    student_probs = student_probs.clamp(min=epsilon)
    kl = (teacher_probs * (teacher_probs.log() - student_probs.log())).sum(dim=1)
    return kl.mean()


def sufficiency_consistency_loss(
    original_logits: Tensor,
    sufficient_logits: Tensor,
    detach_teacher: bool = True,
    epsilon: float = 1e-8,
) -> Tensor:
    """Sufficiency consistency loss.

    L_sufficiency = D_KL( stopgrad(p(x)) || p(x_sufficient) )

    The lesion-sufficient prediction should match the original prediction.
    A high value means the model changed its mind when the background
    was blurred — the lesion alone was not sufficient.

    Args:
        original_logits: [B, C] logits from the original image.
        sufficient_logits: [B, C] logits from the lesion-sufficient image.
        detach_teacher: If True, stop gradients on original_probs.
        epsilon: KL epsilon for stability.

    Returns:
        Scalar loss averaged over batch.
    """
    teacher = torch.softmax(original_logits, dim=1)
    if detach_teacher:
        teacher = teacher.detach()
    return _kl_divergence(teacher, sufficient_logits, epsilon)


def background_consistency_loss(
    original_logits: Tensor,
    swapped_logits: Tensor,
    detach_teacher: bool = True,
    epsilon: float = 1e-8,
) -> Tensor:
    """Background consistency loss.

    L_background = D_KL( stopgrad(p(x)) || p(x_swap) )

    The background-swapped prediction should match the original.
    This penalises reliance on class-correlated background signals.

    Args:
        original_logits: [B, C] logits from the original image.
        swapped_logits: [B, C] logits from the background-swapped image.
        detach_teacher: If True, stop gradients on original_probs.
        epsilon: KL epsilon for stability.

    Returns:
        Scalar loss averaged over batch.
    """
    teacher = torch.softmax(original_logits, dim=1)
    if detach_teacher:
        teacher = teacher.detach()
    return _kl_divergence(teacher, swapped_logits, epsilon)


def necessity_ranking_loss(
    original_logits: Tensor,
    removed_logits: Tensor,
    labels: Tensor,
    margin: float = 0.1,
    confidence_threshold: float = 0.6,
    enabled: bool = True,
) -> tuple[Tensor, Tensor]:
    """Necessity ranking loss with confidence gating.

    L_necessity = max(0, p_y(x_removed) - p_y(x) + m)

    Applied only when:
    - enabled is True (warm-up complete / ramp active)
    - original prediction for true class >= confidence_threshold

    Args:
        original_logits: [B, C] logits from the original image.
        removed_logits: [B, C] logits from the lesion-removed image.
        labels: [B] integer true class indices.
        margin: m (default 0.1).
        confidence_threshold: Minimum p_y(x) for eligibility.
        enabled: If False, returns zero loss for the whole batch.

    Returns:
        (loss, eligible_fraction) — scalar loss and fraction of batch
        that passed the confidence gate (float [0, 1]).
    """
    if not enabled:
        return torch.tensor(0.0, device=original_logits.device), torch.tensor(
            0.0, device=original_logits.device
        )

    original_probs = torch.softmax(original_logits, dim=1)
    removed_probs = torch.softmax(removed_logits, dim=1)

    # p_y for each: gather probability of the true class
    batch_indices = torch.arange(original_logits.size(0), device=original_logits.device)
    p_y_original = original_probs[batch_indices, labels]
    p_y_removed = removed_probs[batch_indices, labels]

    # Confidence gate: only apply to samples where original is confident
    eligible = p_y_original >= confidence_threshold
    eligible_count = eligible.sum().clamp(min=0)

    if eligible_count == 0:
        return torch.tensor(0.0, device=original_logits.device), torch.tensor(
            0.0, device=original_logits.device
        )

    # Hinge loss: max(0, p_y_removed - p_y_original + m)
    per_sample = torch.clamp(p_y_removed - p_y_original + margin, min=0.0)
    masked = per_sample * eligible.float()

    loss = masked.sum() / eligible_count
    fraction = eligible_count.float() / original_logits.size(0)

    return loss, fraction


def compute_causal_losses(
    original_logits: Tensor,
    sufficient_logits: Optional[Tensor],
    removed_logits: Optional[Tensor],
    swapped_logits: Optional[Tensor],
    labels: Tensor,
    config: CausalLossConfig,
    epoch: int = 0,
) -> dict[str, Tensor]:
    """Compute all causal training losses in one call.

    Args:
        original_logits: [B, C] from original image (always required).
        sufficient_logits: [B, C] from lesion-sufficient image or None.
        removed_logits: [B, C] from lesion-removed image or None.
        swapped_logits: [B, C] from background-swapped image or None.
        labels: [B] integer class indices.
        config: CausalLossConfig.
        epoch: Current epoch (0-based), used for warm-up / ramp.

    Returns:
        Dict with keys:
            total_loss: Scalar combined loss.
            ce_loss: Cross-entropy loss value.
            sufficiency_loss: Sufficiency consistency value (or 0).
            background_loss: Background consistency value (or 0).
            necessity_loss: Necessity ranking value (or 0).
            necessity_eligible_fraction: Fraction of batch eligible.
            necessity_weight_current: Active necessity weight.
    """
    variant = config.loss_variant
    device = original_logits.device
    zero = torch.tensor(0.0, device=device)

    # --- Cross-entropy (always) ---
    ce = cross_entropy_loss(original_logits, labels, config.label_smoothing)
    total = config.ce_weight * ce

    # --- Sufficiency ---
    suff = zero
    if variant in ("sufficiency_only", "full") and sufficient_logits is not None:
        suff = sufficiency_consistency_loss(
            original_logits,
            sufficient_logits,
            detach_teacher=config.use_detached_teacher,
            epsilon=config.kl_epsilon,
        )
        total = total + config.sufficiency_weight * suff

    # --- Background ---
    bg = zero
    if variant in ("background_only", "full") and swapped_logits is not None:
        bg = background_consistency_loss(
            original_logits,
            swapped_logits,
            detach_teacher=config.use_detached_teacher,
            epsilon=config.kl_epsilon,
        )
        total = total + config.background_weight * bg

    # --- Necessity ---
    nec = zero
    nec_frac = zero
    nec_weight = zero

    if variant in ("necessity_only", "full") and removed_logits is not None:
        # Warm-up gate
        warmup_done = epoch >= config.necessity_warmup_epochs
        # Ramp weight
        if warmup_done:
            ramp_epochs = max(1, config.necessity_ramp_epochs)
            ramp_progress = min(
                1.0, (epoch - config.necessity_warmup_epochs) / ramp_epochs
            )
            active_weight = config.necessity_weight * ramp_progress
        else:
            active_weight = 0.0
            ramp_progress = 0.0

        nec, nec_frac = necessity_ranking_loss(
            original_logits,
            removed_logits,
            labels,
            margin=config.necessity_margin,
            confidence_threshold=config.necessity_confidence_threshold,
            enabled=warmup_done,
        )
        nec_weight = torch.tensor(active_weight, device=device)
        total = total + nec_weight * nec

    return {
        "total_loss": total,
        "ce_loss": ce,
        "sufficiency_loss": suff,
        "background_loss": bg,
        "necessity_loss": nec,
        "necessity_eligible_fraction": nec_frac,
        "necessity_weight_current": nec_weight,
    }


# ---------------------------------------------------------------------------
# Prediction helpers for monitoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def prediction_entropy(logits: Tensor) -> Tensor:
    """Entropy of predicted distribution in nats (averaged over batch)."""
    probs = torch.softmax(logits, dim=1).clamp(min=1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


def prediction_confidence(logits: Tensor, labels: Optional[Tensor] = None) -> Tensor:
    """Confidence: max probability, or probability of given class.

    Args:
        logits: [B, C].
        labels: Optional [B] for class-specific confidence.

    Returns:
        [B] confidence values.
    """
    probs = torch.softmax(logits, dim=1)
    if labels is not None:
        return probs[torch.arange(logits.size(0), device=logits.device), labels]
    return probs.max(dim=1).values


def divergence_between(
    logits_a: Tensor,
    logits_b: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Symmetric KL (Jensen-Shannon approximation) between two predictions.

    Returns a scalar averaged over the batch — useful for monitoring
    how much the counterfactual changed the prediction.
    """
    p = torch.softmax(logits_a, dim=1).clamp(min=epsilon)
    q = torch.softmax(logits_b, dim=1).clamp(min=epsilon)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum(dim=1) + 0.5 * (q * (q.log() - m.log())).sum(dim=1)
    return js.mean()
