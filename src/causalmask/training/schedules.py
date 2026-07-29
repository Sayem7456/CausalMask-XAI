"""Training schedules for causal loss weights.

Provides:
- LossWeightSchedule: Linear ramp of a loss weight over epochs.
- WarmUpThenRamp: Freeze a loss for warm-up epochs, then ramp linearly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossWeightSchedule:
    """Linearly ramp a loss weight from 0 to target over N epochs.

    Usage:
        sched = LossWeightSchedule(target=0.5, ramp_epochs=10)
        weight = sched(epoch)  # 0.0 at epoch 0, 0.5 at epoch >= 9
    """

    target: float = 0.5
    ramp_epochs: int = 5
    start_epoch: int = 0

    def __call__(self, epoch: int) -> float:
        if self.ramp_epochs <= 0:
            return self.target
        relative = epoch - self.start_epoch
        if relative <= 0:
            return 0.0
        progress = min(1.0, relative / max(1, self.ramp_epochs))
        return self.target * progress

    @property
    def current(self) -> float:
        """Weight after full ramp (for serialisation)."""
        return self.target


@dataclass
class WarmUpThenRamp:
    """Freeze a loss for warm-up epochs, then ramp linearly to target.

    Usage:
        sched = WarmUpThenRamp(target=0.5, warmup_epochs=5, ramp_epochs=5)
        # epoch 0-4: weight=0
        # epoch 5:  weight=0.1
        # epoch 9:  weight=0.5
    """

    target: float = 0.5
    warmup_epochs: int = 5
    ramp_epochs: int = 5

    def __call__(self, epoch: int) -> float:
        relative = epoch - self.warmup_epochs
        if relative < 0:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.target
        progress = min(1.0, relative / max(1, self.ramp_epochs))
        return self.target * progress

    @property
    def current(self) -> float:
        return self.target

    @property
    def active(self) -> bool:
        return self.target > 0.0


def compute_causal_weights(
    config: dict,
    epoch: int,
) -> dict[str, float]:
    """Compute active causal loss weights from a configuration dict.

    Args:
        config: Dict with keys:
            - sufficiency_weight (float)
            - background_weight (float)
            - necessity_weight (float)
            - necessity_warmup_epochs (int)
            - necessity_ramp_epochs (int)
        epoch: Current epoch (0-based).

    Returns:
        Dict with active sufficiency_weight, background_weight,
        necessity_weight.
    """
    sufficiency = float(config.get("sufficiency_weight", 0.5))
    background = float(config.get("background_weight", 0.5))

    necessity_target = float(config.get("necessity_weight", 0.5))
    warmup = int(config.get("necessity_warmup_epochs", 5))
    ramp = int(config.get("necessity_ramp_epochs", 5))

    sched = WarmUpThenRamp(
        target=necessity_target,
        warmup_epochs=warmup,
        ramp_epochs=ramp,
    )
    necessity = sched(epoch)

    return {
        "sufficiency_weight": sufficiency,
        "background_weight": background,
        "necessity_weight": necessity,
    }
