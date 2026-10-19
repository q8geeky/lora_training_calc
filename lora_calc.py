"""Tk desktop UI for reproducible LoRA training workload estimates."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from training_core import (
    TrainingPlan, estimate, epochs_for_target_steps, starting_repeats,
)
from training_profiles import load_profile, save_profile


FIELDS = [
    ("Dataset images", "images", "25"),
    ("Repeats per image", "repeats", "8"),
    ("Epochs", "epochs", "10"),
    ("Batch size (micro-batch)", "batch_size", "2"),
    ("Gradient accumulation", "gradient_accumulation", "1"),
    ("Learning rate", "learning_rate", "0.0001"),
    ("Network rank", "rank", "16"),
    ("Network alpha", "alpha", "16"),
    ("GPU VRAM (GB, informational)", "vram", "8"),
    ("Target optimizer steps (optional)", "target_steps", "1000"),
]


def main():
    root = tk.Tk()
    root.title("LoRA Training Workload Planner")
    root.minsize(610, 620)
    root.columnconfigure(1, weight=1)
    values = {}
    for row, (caption, key, default) in enumerate(FIELDS):
        ttk.Label(root, text=caption).grid(row=row, column=0, sticky="w", padx=14, pady=5)
        variable = tk.StringVar(value=default)
        values[key] = variable
        ttk.Entry(root, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=14, pady=5)

    result = tk.StringVar(value="Enter your trainer settings to estimate updates.")
    note = (
        "VRAM alone cannot determine batch size. Counts assume each epoch flushes\n"
        "its partial batch and accumulation group. Match target changes ONLY epochs."
    )
    ttk.Label(root, text=note, foreground="#555555", justify="left").grid(
        row=len(FIELDS), column=0, columnspan=2, sticky="w", padx=14, pady=9)

    def read_settings():
        plan = TrainingPlan(
            images=int(values["images"].get()),
            repeats=int(values["repeats"].get()),
            epochs=int(values["epochs"].get()),
            batch_size=int(values["batch_size"].get()),
            gradient_accumulation=int(values["gradient_accumulation"].get()),
            learning_rate=float(values["learning_rate"].get()),
            rank=int(values["rank"].get()),
            alpha=int(values["alpha"].get()),
        )
        plan.validate()
        vram = float(values["vram"].get())
        if not math.isfinite(vram) or vram <= 0:
            raise ValueError("GPU VRAM must be a positive, finite number.")
        return plan, vram

    def calculate():
        try:
            plan, _ = read_settings()
            data = estimate(plan)
            result.set(
                f"Samples per epoch: {data.images_per_epoch:,}\n"
                f"Batches per epoch: {data.batches_per_epoch:,}\n"
                f"Optimizer updates per epoch: {data.optimizer_steps_per_epoch:,}\n"
                f"Total optimizer updates: {data.total_optimizer_steps:,}\n"
                f"Total sample exposures: {data.total_image_exposures:,}\n"
                f"Effective batch size: {data.effective_batch_size:,}\n"
                f"Learning rate: {plan.learning_rate:g}   Rank / alpha: {plan.rank} / {plan.alpha}"
            )
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Invalid settings", str(exc))

    def match_target():
        try:
            plan, _ = read_settings()
            target = int(values["target_steps"].get())
            epochs = epochs_for_target_steps(plan, target)
            values["epochs"].set(str(epochs))
            calculate()
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Invalid target", str(exc))

    def suggest():
        try:
            values["repeats"].set(
                str(starting_repeats(int(values["images"].get()))))
            calculate()
        except ValueError as exc:
            messagebox.showerror("Invalid image count", str(exc))

    def export():
        try:
            plan, vram = read_settings()
            path = filedialog.asksaveasfilename(
                defaultextension=".json", filetypes=[("JSON training plan", "*.json")])
            if path:
                save_profile(path, plan, vram)
                messagebox.showinfo("Plan saved", f"Saved your training settings:\n{path}")
        except (OSError, ValueError, OverflowError) as exc:
            messagebox.showerror("Unable to save", str(exc))

    def import_plan():
        path = filedialog.askopenfilename(
            filetypes=[("JSON training plan", "*.json")])
        if not path:
            return
        try:
            plan, vram = load_profile(path)
            import_values = dict(plan.__dict__)
            import_values["vram"] = vram
            for key, value in import_values.items():
                values[key].set(str(value))
            calculate()
        except (OSError, ValueError, OverflowError) as exc:
            messagebox.showerror("Unable to open plan", str(exc))

    tools = ttk.Frame(root)
    tools.grid(row=len(FIELDS)+1, column=0, columnspan=2, pady=10)
    for caption, callback in [
        ("Calculate", calculate),
        ("Match target steps", match_target),
        ("Suggest repeats", suggest),
    ]:
        ttk.Button(tools, text=caption, command=callback).pack(side="left", padx=5)
    files = ttk.Frame(root)
    files.grid(row=len(FIELDS)+2, column=0, columnspan=2, pady=4)
    ttk.Button(files, text="Save settings…", command=export).pack(side="left", padx=7)
    ttk.Button(files, text="Open settings…", command=import_plan).pack(side="left", padx=7)
    ttk.Label(root, textvariable=result, justify="left").grid(
        row=len(FIELDS)+3, column=0, columnspan=2, sticky="w", padx=15, pady=10)
    calculate()
    root.mainloop()


if __name__ == "__main__":
    main()
