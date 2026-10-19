"""Portable, validated training-plan profiles with atomic JSON writes."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
from math import isfinite
import os
from pathlib import Path
import tempfile

from training_core import TrainingPlan

SCHEMA_VERSION = 1
PLAN_FIELDS = {field.name for field in fields(TrainingPlan)}


def _validate_vram(vram_gb: float) -> float:
    try:
        number = float(vram_gb)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("VRAM must be a positive finite number of GB.") from exc
    if not isfinite(number) or number <= 0:
        raise ValueError("VRAM must be a positive finite number of GB.")
    return number


def profile_document(plan: TrainingPlan, vram_gb: float) -> dict:
    plan.validate()
    return {
        "format": "lora-training-plan",
        "version": SCHEMA_VERSION,
        "settings": asdict(plan),
        "vram_gb": _validate_vram(vram_gb),
        "note": "Estimate only. These are not model- or trainer-specific recommended settings.",
    }


def parse_profile_document(value: object) -> tuple[TrainingPlan, float]:
    if not isinstance(value, dict) or value.get("format") != "lora-training-plan" or value.get("version") != SCHEMA_VERSION:
        raise ValueError("Unsupported LoRA training plan format/version.")
    settings = value.get("settings")
    if not isinstance(settings, dict) or set(settings) != PLAN_FIELDS:
        raise ValueError("Profile must contain exactly the supported training settings.")
    # Dataclasses do not enforce runtime types; validate field kinds explicitly.
    for name in PLAN_FIELDS - {"learning_rate"}:
        x = settings[name]
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(f"{name} must be a whole number.")
    try:
        learning_rate = float(settings["learning_rate"])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Learning rate must be a positive finite number.") from exc
    plan = TrainingPlan(**{**settings, "learning_rate": learning_rate})
    plan.validate()
    return plan, _validate_vram(value.get("vram_gb"))


def save_profile(path: Path | str, plan: TrainingPlan, vram_gb: float) -> None:
    path = Path(path)
    document = profile_document(plan, vram_gb)
    # A failed save leaves an earlier valid profile untouched.
    fd, tmp_name = tempfile.mkstemp(dir=path.resolve().parent, prefix=".lora_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_profile(path: Path | str) -> tuple[TrainingPlan, float]:
    with Path(path).open("r", encoding="utf-8") as source:
        return parse_profile_document(json.load(source))
