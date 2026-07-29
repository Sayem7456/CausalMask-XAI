"""Unit tests for causal training losses and schedules."""

import math

import pytest
import torch

from causalmask.training.losses import (
    CausalLossConfig,
    _kl_divergence,
    background_consistency_loss,
    compute_causal_losses,
    cross_entropy_loss,
    divergence_between,
    necessity_ranking_loss,
    prediction_confidence,
    prediction_entropy,
    sufficiency_consistency_loss,
)
from causalmask.training.schedules import (
    LossWeightSchedule,
    WarmUpThenRamp,
    compute_causal_weights,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _logits_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """Convert probabilities to logits (inverse softmax up to an additive constant)."""
    return probs.log()


def _assert_finite(t: torch.Tensor, name: str = "tensor"):
    assert torch.isfinite(t).all(), f"{name} contains non-finite values"


def _assert_non_negative(t: torch.Tensor, name: str = "tensor"):
    assert (t >= 0).all(), f"{name} contains negative values"


# ---------------------------------------------------------------------------
# cross-entropy
# ---------------------------------------------------------------------------


def test_ce_perfect_prediction():
    logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    labels = torch.tensor([1, 0])
    loss = cross_entropy_loss(logits, labels)
    _assert_finite(loss, "ce_loss")
    assert loss.item() < 0.01


def test_ce_wrong_prediction():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([1, 0])
    loss = cross_entropy_loss(logits, labels)
    _assert_finite(loss, "ce_loss")
    assert loss.item() > 1.0


def test_ce_label_smoothing():
    logits = torch.tensor([[0.0, 10.0]])
    labels = torch.tensor([1])
    no_smooth = cross_entropy_loss(logits, labels, label_smoothing=0.0)
    with_smooth = cross_entropy_loss(logits, labels, label_smoothing=0.1)
    assert with_smooth > no_smooth


def test_ce_gradient_flow():
    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    loss = cross_entropy_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert (logits.grad != 0).any()


# ---------------------------------------------------------------------------
# KL divergence (internal helper)
# ---------------------------------------------------------------------------


def test_kl_identical_distributions():
    probs = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    logits = _logits_from_probs(probs)
    kl = _kl_divergence(probs, logits)
    _assert_finite(kl, "kl")
    assert kl.item() < 1e-6


def test_kl_different_distributions_positive():
    teacher = torch.tensor([[0.9, 0.1]])
    student_logits = _logits_from_probs(torch.tensor([[0.1, 0.9]]))
    kl = _kl_divergence(teacher, student_logits)
    _assert_finite(kl, "kl")
    assert kl.item() > 0.5


def test_kl_batch_averaging():
    teacher = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    student_logits = _logits_from_probs(torch.tensor([[0.1, 0.9], [0.8, 0.2]]))
    kl = _kl_divergence(teacher, student_logits)
    _assert_finite(kl, "kl")


# ---------------------------------------------------------------------------
# sufficiency consistency loss
# ---------------------------------------------------------------------------


def test_sufficiency_zero_loss_identical():
    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    loss = sufficiency_consistency_loss(logits, logits)
    _assert_finite(loss, "suff")
    assert loss.item() < 1e-5


def test_sufficiency_positive_loss_different():
    original = torch.tensor([[1.0, 3.0], [2.0, 1.0]])
    sufficient = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    loss = sufficiency_consistency_loss(original, sufficient)
    _assert_finite(loss, "suff")
    assert loss.item() > 0.0


def test_sufficiency_detached_teacher():
    original = torch.tensor([[1.0, 3.0]], requires_grad=True)
    sufficient = torch.tensor([[2.0, 1.0]], requires_grad=True)
    loss = sufficiency_consistency_loss(original, sufficient, detach_teacher=True)
    loss.backward()
    # Teacher was detached — original should have no gradient
    assert original.grad is None or (original.grad == 0).all()
    assert sufficient.grad is not None
    assert (sufficient.grad != 0).any()


def test_sufficiency_non_detached_teacher():
    original = torch.tensor([[1.0, 3.0]], requires_grad=True)
    sufficient = torch.tensor([[2.0, 1.0]], requires_grad=True)
    loss = sufficiency_consistency_loss(original, sufficient, detach_teacher=False)
    loss.backward()
    # Both should have gradients
    assert original.grad is not None and (original.grad != 0).any()
    assert sufficient.grad is not None and (sufficient.grad != 0).any()


def test_sufficiency_gradient_finite():
    original = torch.tensor([[1.0, 3.0], [2.0, 1.0]], requires_grad=True)
    sufficient = torch.tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
    loss = sufficiency_consistency_loss(original, sufficient)
    loss.backward()
    assert sufficient.grad is not None
    _assert_finite(sufficient.grad, "sufficient_grad")


# ---------------------------------------------------------------------------
# background consistency loss
# ---------------------------------------------------------------------------


def test_background_zero_loss_identical():
    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    loss = background_consistency_loss(logits, logits)
    _assert_finite(loss, "bg")
    assert loss.item() < 1e-5


def test_background_positive_loss_different():
    original = torch.tensor([[1.0, 3.0]])
    swapped = torch.tensor([[3.0, 1.0]])
    loss = background_consistency_loss(original, swapped)
    _assert_finite(loss, "bg")
    assert loss.item() > 0.0


def test_background_detached_teacher():
    original = torch.tensor([[1.0, 3.0]], requires_grad=True)
    swapped = torch.tensor([[2.0, 1.0]], requires_grad=True)
    loss = background_consistency_loss(original, swapped, detach_teacher=True)
    loss.backward()
    assert original.grad is None or (original.grad == 0).all()
    assert swapped.grad is not None and (swapped.grad != 0).any()


# ---------------------------------------------------------------------------
# necessity ranking loss
# ---------------------------------------------------------------------------


def test_necessity_zero_when_removed_lower():
    """p_y(x_removed) < p_y(x) => max(0, negative + m) may still be 0 if m small."""
    # p(x) = 0.9, p(removed) = 0.3, m=0.1 => max(0, 0.3-0.9+0.1) = max(0, -0.5) = 0
    original = _logits_from_probs(torch.tensor([[0.1, 0.9]]))
    removed = _logits_from_probs(torch.tensor([[0.7, 0.3]]))
    labels = torch.tensor([1])
    loss, frac = necessity_ranking_loss(
        original, removed, labels, margin=0.1, confidence_threshold=0.5
    )
    _assert_finite(loss, "nec")
    assert loss.item() == 0.0
    assert frac.item() == 1.0  # all eligible (0.9 >= 0.5)


def test_necessity_positive_when_removed_higher():
    """p_y(x_removed) > p_y(x) => penalty applies."""
    # p(x) = 0.6, p(removed) = 0.8, m=0.0 => max(0, 0.8-0.6+0.0) = 0.2
    original = _logits_from_probs(torch.tensor([[0.4, 0.6]]))
    removed = _logits_from_probs(torch.tensor([[0.2, 0.8]]))
    labels = torch.tensor([1])
    loss, frac = necessity_ranking_loss(
        original, removed, labels, margin=0.0, confidence_threshold=0.5
    )
    _assert_finite(loss, "nec")
    assert abs(loss.item() - 0.2) < 1e-4
    assert frac.item() == 1.0


def test_necessity_margin_effect():
    """Larger margin => more penalty."""
    original = _logits_from_probs(torch.tensor([[0.4, 0.6]]))
    removed = _logits_from_probs(torch.tensor([[0.3, 0.7]]))
    labels = torch.tensor([1])
    # m=0.0: max(0, 0.7-0.6) = 0.1
    # m=0.2: max(0, 0.7-0.6+0.2) = 0.3
    loss_0, _ = necessity_ranking_loss(original, removed, labels, margin=0.0)
    loss_2, _ = necessity_ranking_loss(original, removed, labels, margin=0.2)
    assert loss_0.item() > 0.0
    assert loss_2.item() > loss_0.item()


def test_necessity_disabled_returns_zero():
    original = _logits_from_probs(torch.tensor([[0.1, 0.9]]))
    removed = _logits_from_probs(torch.tensor([[0.0, 1.0]]))
    labels = torch.tensor([1])
    loss, frac = necessity_ranking_loss(
        original, removed, labels, enabled=False
    )
    assert loss.item() == 0.0
    assert frac.item() == 0.0


def test_necessity_confidence_gating():
    """Sample below confidence threshold is excluded."""
    # p(x) for true class = 0.4 (< 0.6 threshold) → ineligible
    # p(x) for true class = 0.9 (>= 0.6) → eligible
    original = _logits_from_probs(torch.tensor([[0.6, 0.4], [0.1, 0.9]]))
    removed = _logits_from_probs(torch.tensor([[0.2, 0.8], [0.1, 0.9]]))
    labels = torch.tensor([1, 1])
    loss, frac = necessity_ranking_loss(
        original, removed, labels,
        margin=0.0, confidence_threshold=0.6,
    )
    # Only the second sample is eligible
    assert abs(frac.item() - 0.5) < 1e-4
    # First sample: ineligible.  Second: max(0, 0.9-0.9) = 0
    assert loss.item() == 0.0


def test_necessity_no_eligible_samples():
    """All samples below threshold => zero loss, zero fraction."""
    original = _logits_from_probs(torch.tensor([[0.7, 0.3], [0.6, 0.4]]))
    removed = _logits_from_probs(torch.tensor([[0.2, 0.8], [0.1, 0.9]]))
    labels = torch.tensor([1, 1])
    loss, frac = necessity_ranking_loss(
        original, removed, labels,
        margin=0.0, confidence_threshold=0.8,
    )
    assert loss.item() == 0.0
    assert frac.item() == 0.0


def test_necessity_gradient_flow():
    original = torch.tensor([[1.0, 2.0]], requires_grad=True)
    removed = torch.tensor([[0.5, 3.0]], requires_grad=True)
    labels = torch.tensor([1])
    loss, _ = necessity_ranking_loss(
        original, removed, labels, margin=0.0, confidence_threshold=0.4
    )
    loss.backward()
    assert original.grad is not None
    assert removed.grad is not None
    assert (original.grad != 0).any() or (removed.grad != 0).any()
    _assert_finite(original.grad, "original_grad")
    _assert_finite(removed.grad, "removed_grad")


# ---------------------------------------------------------------------------
# compute_causal_losses (integration)
# ---------------------------------------------------------------------------


def test_full_causal_objective_returns_all_keys():
    config = CausalLossConfig(loss_variant="full")
    B, C = 4, 2
    original = torch.randn(B, C)
    sufficient = torch.randn(B, C)
    removed = torch.randn(B, C)
    swapped = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))

    result = compute_causal_losses(
        original, sufficient, removed, swapped, labels, config, epoch=10
    )
    expected_keys = {
        "total_loss", "ce_loss", "sufficiency_loss", "background_loss",
        "necessity_loss", "necessity_eligible_fraction", "necessity_weight_current",
    }
    assert set(result.keys()) == expected_keys
    for k, v in result.items():
        _assert_finite(v, k)


def test_ce_only_variant():
    config = CausalLossConfig(loss_variant="ce_only")
    original = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    result = compute_causal_losses(
        original, None, None, None, labels, config, epoch=0
    )
    assert result["sufficiency_loss"].item() == 0.0
    assert result["background_loss"].item() == 0.0
    assert result["necessity_loss"].item() == 0.0


def test_sufficiency_only_variant():
    config = CausalLossConfig(loss_variant="sufficiency_only")
    original = torch.randn(4, 2)
    sufficient = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    result = compute_causal_losses(
        original, sufficient, None, None, labels, config, epoch=0
    )
    assert result["sufficiency_loss"].item() > 0.0
    assert result["background_loss"].item() == 0.0
    assert result["necessity_loss"].item() == 0.0


def test_necessity_warmup_zero_loss():
    """Necessity loss should be 0 during warm-up."""
    config = CausalLossConfig(
        loss_variant="full",
        necessity_warmup_epochs=5,
        necessity_weight=1.0,
    )
    original = _logits_from_probs(torch.tensor([[0.3, 0.7]]))
    removed = _logits_from_probs(torch.tensor([[0.2, 0.8]]))
    labels = torch.tensor([1])
    result = compute_causal_losses(
        original, None, removed, None, labels, config, epoch=2
    )
    assert result["necessity_loss"].item() == 0.0
    assert result["necessity_weight_current"].item() == 0.0


def test_necessity_after_warmup_active():
    config = CausalLossConfig(
        loss_variant="full",
        necessity_warmup_epochs=5,
        necessity_ramp_epochs=5,
        necessity_weight=1.0,
        necessity_confidence_threshold=0.0,
    )
    original = _logits_from_probs(torch.tensor([[0.3, 0.7]]))
    removed = _logits_from_probs(torch.tensor([[0.2, 0.8]]))
    labels = torch.tensor([1])
    # epoch 10 is well past warmup+ramp
    result = compute_causal_losses(
        original, None, removed, None, labels, config, epoch=10
    )
    assert result["necessity_loss"].item() >= 0.0
    assert result["necessity_weight_current"].item() > 0.0


def test_necessity_ramp_increases():
    config = CausalLossConfig(
        loss_variant="full",
        necessity_warmup_epochs=5,
        necessity_ramp_epochs=5,
        necessity_weight=1.0,
        necessity_confidence_threshold=0.0,
    )
    original = torch.randn(4, 2)
    removed = torch.randn(4, 2)
    labels = torch.zeros(4, dtype=torch.long)

    r1 = compute_causal_losses(original, None, removed, None, labels, config, epoch=6)
    r2 = compute_causal_losses(original, None, removed, None, labels, config, epoch=9)
    # epoch 9 weight >= epoch 6 weight (ramp is monotonic)
    assert r2["necessity_weight_current"] >= r1["necessity_weight_current"]


def test_total_loss_is_sum_of_parts():
    config = CausalLossConfig(
        loss_variant="full",
        ce_weight=1.0,
        sufficiency_weight=2.0,
        background_weight=0.0,
        necessity_weight=0.0,
    )
    original = torch.randn(4, 2)
    sufficient = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    result = compute_causal_losses(
        original, sufficient, None, None, labels, config, epoch=10
    )
    expected = result["ce_loss"] + 2.0 * result["sufficiency_loss"]
    assert abs(result["total_loss"].item() - expected.item()) < 1e-5


def test_total_loss_finite():
    config = CausalLossConfig(loss_variant="full")
    original = torch.randn(4, 2)
    sufficient = torch.randn(4, 2)
    removed = torch.randn(4, 2)
    swapped = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    result = compute_causal_losses(
        original, sufficient, removed, swapped, labels, config, epoch=10
    )
    _assert_finite(result["total_loss"], "total_loss")
    assert not torch.isnan(result["total_loss"])


# ---------------------------------------------------------------------------
# prediction monitoring helpers
# ---------------------------------------------------------------------------


def test_prediction_entropy_uniform():
    logits = torch.zeros(1, 4)
    ent = prediction_entropy(logits)
    _assert_finite(ent, "entropy")
    assert abs(ent.item() - math.log(4)) < 1e-4


def test_prediction_entropy_confident():
    logits = torch.tensor([[0.0, 10.0]])
    ent = prediction_entropy(logits)
    assert ent.item() < 0.1


def test_prediction_confidence_max():
    logits = torch.tensor([[0.0, 10.0]])
    conf = prediction_confidence(logits)
    assert conf.item() > 0.99


def test_prediction_confidence_class():
    logits = torch.tensor([[0.0, 10.0], [5.0, 0.0]])
    labels = torch.tensor([1, 0])
    conf = prediction_confidence(logits, labels)
    assert conf[0].item() > 0.99
    assert conf[1].item() > 0.99


def test_divergence_between_identical():
    logits = torch.randn(4, 2)
    div = divergence_between(logits, logits)
    assert div.item() < 1e-6


def test_divergence_between_different():
    a = torch.tensor([[0.0, 20.0]])
    b = torch.tensor([[20.0, 0.0]])
    div = divergence_between(a, b)
    assert div.item() > 0.5


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------


def test_loss_weight_schedule_zero_start():
    sched = LossWeightSchedule(target=1.0, ramp_epochs=10, start_epoch=0)
    assert sched(0) == 0.0


def test_loss_weight_schedule_full_ramp():
    sched = LossWeightSchedule(target=1.0, ramp_epochs=5, start_epoch=0)
    assert sched(4) == 0.8
    assert sched(5) == 1.0
    assert sched(10) == 1.0


def test_loss_weight_schedule_no_ramp():
    sched = LossWeightSchedule(target=0.5, ramp_epochs=0)
    assert sched(0) == 0.5


def test_loss_weight_schedule_start_offset():
    sched = LossWeightSchedule(target=1.0, ramp_epochs=5, start_epoch=3)
    assert sched(0) == 0.0
    assert sched(2) == 0.0
    assert sched(3) == 0.0
    assert sched(5) == 0.4
    assert sched(8) == 1.0


def test_warmup_then_ramp_frozen_during_warmup():
    sched = WarmUpThenRamp(target=1.0, warmup_epochs=5, ramp_epochs=5)
    for e in range(5):
        assert sched(e) == 0.0, f"epoch {e} should be 0 during warm-up"


def test_warmup_then_ramp_after_warmup():
    sched = WarmUpThenRamp(target=1.0, warmup_epochs=5, ramp_epochs=5)
    assert sched(5) == 0.0  # first ramp epoch
    assert sched(7) == 0.4
    assert sched(9) == 0.8
    assert sched(10) == 1.0
    assert sched(20) == 1.0


def test_warmup_then_ramp_no_ramp():
    sched = WarmUpThenRamp(target=0.5, warmup_epochs=3, ramp_epochs=0)
    assert sched(0) == 0.0
    assert sched(2) == 0.0
    assert sched(3) == 0.5
    assert sched(10) == 0.5


def test_compute_causal_weights():
    config = {
        "sufficiency_weight": 0.7,
        "background_weight": 0.3,
        "necessity_weight": 0.5,
        "necessity_warmup_epochs": 3,
        "necessity_ramp_epochs": 2,
    }
    # During warm-up
    w0 = compute_causal_weights(config, epoch=0)
    assert w0["sufficiency_weight"] == 0.7
    assert w0["background_weight"] == 0.3
    assert w0["necessity_weight"] == 0.0

    # After full ramp
    w5 = compute_causal_weights(config, epoch=5)
    assert w5["necessity_weight"] == 0.5


# ---------------------------------------------------------------------------
# loss variant enum validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["ce_only", "necessity_only", "sufficiency_only", "background_only", "full"])
def test_all_variants_produce_finite_total(variant):
    config = CausalLossConfig(loss_variant=variant)
    original = torch.randn(4, 2)
    sufficient = torch.randn(4, 2)
    removed = torch.randn(4, 2)
    swapped = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    result = compute_causal_losses(
        original, sufficient, removed, swapped, labels, config, epoch=10
    )
    _assert_finite(result["total_loss"], f"total_loss_{variant}")
    _assert_non_negative(result["total_loss"], f"total_loss_{variant}")
