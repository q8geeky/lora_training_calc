"""Interactive front-end for the deterministic LoRA workload calculator."""
import tkinter as tk
from tkinter import ttk, messagebox
from training_core import TrainingPlan, estimate, starting_repeats


def main():
    root = tk.Tk()
    root.title("LoRA Training Planner")
    root.minsize(540, 560)
    root.columnconfigure(1, weight=1)
    fields = [
        ("Dataset images", "images", "25"),
        ("Repeats per image", "repeats", "8"),
        ("Epochs", "epochs", "10"),
        ("Batch size (micro-batch)", "batch_size", "2"),
        ("Gradient accumulation", "gradient_accumulation", "1"),
        ("Learning rate", "learning_rate", "0.0001"),
        ("Network rank", "rank", "16"),
        ("Network alpha", "alpha", "16"),
        ("GPU VRAM (GB, informational)", "vram", "8"),
    ]
    vars_ = {}
    for row,(caption,key,default) in enumerate(fields):
        ttk.Label(root,text=caption).grid(row=row,column=0,sticky="w",padx=14,pady=7)
        variable=tk.StringVar(value=default)
        vars_[key]=variable
        ttk.Entry(root,textvariable=variable).grid(row=row,column=1,sticky="ew",padx=14,pady=7)
    result=tk.StringVar(value="Enter the actual settings used by your trainer.")
    note=("VRAM cannot determine batch size on its own. Estimates assume each epoch\n"
          "flushes a partial batch/gradient accumulation group.")
    ttk.Label(root,text=note,foreground="#555555",justify="left").grid(
        row=len(fields),column=0,columnspan=2,sticky="w",padx=14,pady=8)

    def calculate():
        try:
            plan=TrainingPlan(
                images=int(vars_["images"].get()),
                repeats=int(vars_["repeats"].get()),
                epochs=int(vars_["epochs"].get()),
                batch_size=int(vars_["batch_size"].get()),
                gradient_accumulation=int(vars_["gradient_accumulation"].get()),
                learning_rate=float(vars_["learning_rate"].get()),
                rank=int(vars_["rank"].get()),
                alpha=int(vars_["alpha"].get()),
            )
            vram=float(vars_["vram"].get())
            if not (0 < vram < float("inf")):
                raise ValueError("VRAM must be a positive finite number.")
            data=estimate(plan)
            result.set(
                f"Samples per epoch: {data.images_per_epoch:,}\n"
                f"Batches per epoch: {data.batches_per_epoch:,}\n"
                f"Optimizer updates per epoch: {data.optimizer_steps_per_epoch:,}\n"
                f"Total optimizer updates: {data.total_optimizer_steps:,}\n"
                f"Total sample exposures: {data.total_image_exposures:,}\n"
                f"Effective batch size: {data.effective_batch_size:,}\n"
                f"Learning rate: {plan.learning_rate:g}\n"
                f"LoRA rank / alpha: {plan.rank} / {plan.alpha}"
            )
        except (ValueError,OverflowError) as exc:
            messagebox.showerror("Invalid settings",str(exc))

    def suggest():
        try:
            vars_["repeats"].set(str(starting_repeats(int(vars_["images"].get()))))
        except ValueError as exc:
            messagebox.showerror("Invalid image count",str(exc))

    controls=ttk.Frame(root)
    controls.grid(row=len(fields)+1,column=0,columnspan=2,pady=12)
    ttk.Button(controls,text="Calculate",command=calculate).pack(side="left",padx=8)
    ttk.Button(controls,text="Suggest repeats",command=suggest).pack(side="left",padx=8)
    ttk.Label(root,textvariable=result,justify="left").grid(
        row=len(fields)+2,column=0,columnspan=2,sticky="w",padx=15,pady=10)
    calculate()
    root.mainloop()


if __name__=="__main__":
    main()
