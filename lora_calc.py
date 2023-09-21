import tkinter as tk
from tkinter import ttk, messagebox

def calculate_repeats(num_images):
    """Calculate the number of repeats based on the number of dataset images."""
    if num_images <= 20:
        return 40
    elif num_images >= 100:
        return 10
    else:
        # For images between 20 and 100, we'll use an average of 70 and 100.
        return (70 + 100) // 2

def calculate_batch_size(vram):
    """Calculate the batch size based on VRAM."""
    # Directly return the VRAM as batch size (assuming VRAM is a good proxy for batch size).
    return int(vram)

def calculate_epoch():
    """Calculate the epoch, fixed between 5 and 10."""
    # Return the average of 5 and 10 as a general guideline.
    return (5 + 10) // 2

def calculate_learning_rate(batch_size):
    """Calculate the learning rate based on the batch size."""
    # Learning rate is proportional to the batch size.
    return 0.0002 * batch_size

def calculate_network_rank_and_alpha(num_images):
    """Calculate the network rank and alpha based on the number of dataset images."""
    # Determine network rank based on the number of images.
    network_rank = 64 if num_images < 25 else 128
    # Network alpha is half the network rank.
    network_alpha = network_rank // 2
    return network_rank, network_alpha

def calculate_steps(num_images, repeats):
    """Calculate the total number of steps."""
    return num_images * repeats

def calculate_regularization_images(num_images, repeats):
    """Calculate the number of regularization dataset images."""
    # 2 regularization images for every dataset image.
    return (num_images * repeats) * 2

def calculate_max_training_step(num_images, repeats, batch_size, epoch):
    """Calculate the maximum training step."""
    return ((num_images * repeats) * 1) / (batch_size / (1 * epoch * batch_size))

def calculate_parameters():
    """Calculate and display all the parameters."""
    try:
        vram = float(vram_entry.get())
        num_images = int(num_images_entry.get())
        
        # Calculations
        repeats = calculate_repeats(num_images)
        batch_size = calculate_batch_size(vram)
        epoch = calculate_epoch()
        learning_rate = calculate_learning_rate(batch_size)
        network_rank, network_alpha = calculate_network_rank_and_alpha(num_images)
        steps = calculate_steps(num_images, repeats)
        regularization_images = calculate_regularization_images(num_images, repeats)
        max_training_step = calculate_max_training_step(num_images, repeats, batch_size, epoch)
        
        # Display results
        results_text.set(f"Repeats: {repeats}\nBatch Size: {batch_size}\nEpoch: {epoch}\nLearning Rate: {learning_rate}\nNetwork Rank: {network_rank}\nNetwork Alpha: {network_alpha}\nSteps: {steps}\nRegularization Dataset Images: {regularization_images}\nMax Training Step: {max_training_step}")
    
    except ValueError:
        messagebox.showerror("Error", "Please enter valid values.")
    except ZeroDivisionError:
        messagebox.showerror("Error", "Division by zero detected. Please enter valid values.")

# Create main window
root = tk.Tk()
root.title("Lora Training Parameters Calculator")

# Create and place labels, entries, and button
ttk.Label(root, text="VRAM:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
vram_entry = ttk.Entry(root)
vram_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(root, text="Number of Dataset Images:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
num_images_entry = ttk.Entry(root)
num_images_entry.grid(row=1, column=1, padx=10, pady=5)

calculate_button = ttk.Button(root, text="Calculate", command=calculate_parameters)
calculate_button.grid(row=2, column=0, columnspan=2, pady=10)

results_text = tk.StringVar()
results_label = ttk.Label(root, textvariable=results_text)
results_label.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()
