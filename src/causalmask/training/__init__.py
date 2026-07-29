"""Training modules for CausalMask-XAI.

Includes standard training engine, causal losses, weight schedules,
checkpointing, and a causal trainer.
"""

from causalmask.training.engine import Trainer, TrainingConfig
from causalmask.training.checkpointing import (
    Checkpoint,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    capture_rng_states,
    restore_rng_states,
)
from causalmask.training.losses import (
    CausalLossConfig,
    cross_entropy_loss,
    sufficiency_consistency_loss,
    background_consistency_loss,
    necessity_ranking_loss,
    compute_causal_losses,
)
from causalmask.training.schedules import (
    LossWeightSchedule,
    WarmUpThenRamp,
    compute_causal_weights,
)
from causalmask.training.causal_trainer import CausalTrainer
