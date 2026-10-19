# LoRA Training Workload Planner

A desktop calculator for **training workload arithmetic**. It is not a GPU memory estimator, hyperparameter tuner, or an automatic training-quality predictor.

## Run

Install Python 3.10+ with Tkinter. No pip dependencies are required.

```sh
python lora_calc.py
```

On Windows, double-click `START_LORA_CALC.bat`.

## New features

- Enter dataset images, repeats, epochs, micro-batch size and gradient accumulation.
- Show per-epoch samples, batches, optimizer updates, total sample exposures and effective batch size.
- **Match target steps** computes the *minimum integer number of epochs* needed for the desired number of optimizer steps. This changes only `epochs`, never training quality settings.
- **Save settings…** and **Open settings…** use versioned JSON plan files with input validation and atomic writes. Plans are portable between machines.
- Rank, alpha, learning rate and VRAM remain explicit user inputs.

## Arithmetic

Given `I` images, `R` repeats, `E` epochs, `B` micro-batch, and `G` accumulation:

```text
samples_per_epoch     = I * R
batches_per_epoch     = ceil(samples_per_epoch / B)
optimizer_steps_epoch = ceil(batches_per_epoch / G)
optimizer_steps_total = optimizer_steps_epoch * E
effective_batch_size  = B * G
sample_exposures      = I * R * E
```

For desired optimizer steps `S`:

```text
required_epochs = ceil(S / optimizer_steps_epoch)
```

**Assumptions:** each epoch flushes a partial final batch and accumulation group; no warm-up-specific or prior stopping logic; no regularization images added unless included in the image count. Some trainers handle epoch boundaries differently—always compare with the actual trainer logs.

The **Suggest repeats** button uses a bounded heuristic targeting roughly 200 exposures per epoch. It does not necessarily improve results.

GPU VRAM alone cannot tell you what batch size fits; architecture, resolution, precision, optimizer and checkpointing affect memory.

## Profile format

`Save settings` stores `format`, `version`, the eight training inputs, informational `vram_gb`, and a note. Version is currently `1`. Unknown formats/versions and invalid values are rejected without changing the current GUI settings.

## Automated tests

```sh
python -m unittest discover -s tests -v
```

Pure calculation and profile code import without launching a Tk window. GitHub Actions tests Python 3.10/3.12/3.13.

## Safety

These estimates are **informational only**. They never launch training, access your models, read your dataset or configure GPU software.
