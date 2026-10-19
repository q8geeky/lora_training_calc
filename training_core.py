"""Deterministic LoRA training workload estimates.

Training-step counts are arithmetic, not a promise of quality or VRAM fit.
Different trainers may have different rules for regularization/class images.
"""
from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class TrainingPlan:
    images: int
    repeats: int
    epochs: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    rank: int
    alpha: int

    def validate(self) -> None:
        positive = {
            "Images": self.images,
            "Repeats": self.repeats,
            "Epochs": self.epochs,
            "Batch size": self.batch_size,
            "Gradient accumulation": self.gradient_accumulation,
            "Network rank": self.rank,
            "Network alpha": self.alpha,
        }
        for label, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive whole number.")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be a positive, finite number.")
        if self.alpha > self.rank * 4:
            raise ValueError("Network alpha is unusually large relative to rank; check the input.")


@dataclass(frozen=True)
class TrainingEstimate:
    images_per_epoch: int
    batches_per_epoch: int
    optimizer_steps_per_epoch: int
    total_optimizer_steps: int
    total_image_exposures: int
    effective_batch_size: int


def estimate(plan: TrainingPlan) -> TrainingEstimate:
    """Estimate update steps for a trainer that rounds at each epoch boundary.

    Each epoch sees images * repeats samples. A partial final batch and a
    partial final gradient-accumulation group still generate an update.
    """
    plan.validate()
    images_per_epoch = plan.images * plan.repeats
    batches_per_epoch = ceil(images_per_epoch / plan.batch_size)
    steps_per_epoch = ceil(batches_per_epoch / plan.gradient_accumulation)
    return TrainingEstimate(
        images_per_epoch=images_per_epoch,
        batches_per_epoch=batches_per_epoch,
        optimizer_steps_per_epoch=steps_per_epoch,
        total_optimizer_steps=steps_per_epoch * plan.epochs,
        total_image_exposures=images_per_epoch * plan.epochs,
        effective_batch_size=plan.batch_size * plan.gradient_accumulation,
    )


def starting_repeats(images: int, target_exposures_per_epoch: int = 200) -> int:
    """Optional *heuristic*, not a recommendation for a particular trainer."""
    if images < 1 or target_exposures_per_epoch < 1:
        raise ValueError("Image count and target exposures must be positive.")
    return min(40, max(1, ceil(target_exposures_per_epoch / images)))
