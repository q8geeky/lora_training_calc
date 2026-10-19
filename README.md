# LoRA Training Workload Calculator

A lightweight desktop calculator for **training-step arithmetic**. It no longer treats GPU VRAM as batch size or invents training recommendations from VRAM.

## Run

Python 3.10+ with Tkinter, no pip dependencies:

```sh
python lora_calc.py
```

Windows: double-click `START_LORA_CALC.bat`.

## Inputs and equations

Images = I, repeats = R, epochs = E, batch = B, gradient accumulation = G.

```
samples_per_epoch = I * R
batches_per_epoch = ceil(samples_per_epoch / B)
steps_per_epoch = ceil(batches_per_epoch / G)
total_optimizer_steps = steps_per_epoch * E
total_exposures = samples_per_epoch * E
effective_batch = B * G
```

Each epoch flushes remaining partial batches/accumulation groups. Your actual training library may differ. Learning rate, rank, and alpha remain explicitly user-entered because they cannot be derived from dataset count or VRAM alone. VRAM is informational only.

The "Suggest repeats" button estimates repeats to reach ~200 sample exposures per epoch, capped at 40. It is a heuristic, **not a universal optimal setting**.

## Tests

```sh
python -m unittest discover -s tests -v
```

Pure arithmetic lives in `training_core.py` and does not create windows when imported.
