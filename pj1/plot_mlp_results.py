import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import gzip
from struct import unpack
import pickle # Need pickle to load models saved by runner

# Import necessary components from mynn
from mynn.models import Model_MLP
from mynn.metric import accuracy

# plt.style.use(['science'])

RESULTS_DIR = 'experiment_results/mlp'
FIGS_DIR = 'figs/mlp'
MODELS_DIR = 'saved_models/mlp' # Directory where models are saved
os.makedirs(FIGS_DIR, exist_ok=True)
# Note: We assume MODELS_DIR exists and contains the saved models

MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']

# --- Constants for MLP Model (matching run_mlp_experiments.py) ---
MLP_SIZE_LIST = [784, 600, 10]
MLP_ACT_FUNC = 'ReLU'
MLP_LAMBDA_LIST = [1e-4, 1e-4]
# ---

# --- MNIST Test Data Loading ---
def load_mnist_test_data():
    """Loads and preprocesses the MNIST test dataset."""
    data_dir = 'dataset/MNIST'
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    if not (os.path.exists(test_images_path) and os.path.exists(test_labels_path)):
        print(f"Error: MNIST test data not found in {data_dir}")
        return None, None

    print(f"Loading test images: {test_images_path}")
    with gzip.open(test_images_path, 'rb') as f:
        magic, num_images, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)

    print(f"Loading test labels: {test_labels_path}")
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num_labels = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    # Normalize pixel values from [0, 255] to [0, 1]
    test_imgs = test_imgs / 255.0

    print(f"Test samples loaded: {test_imgs.shape[0]}")
    return test_imgs, test_labs

# --- Load Experiment Summary ---
def load_results():
    """Loads only the summary file to get experiment names."""
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return None, None
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    histories = {}
    for exp_name in summary.keys():
        history_path = os.path.join(RESULTS_DIR, f"{exp_name}_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                histories[exp_name] = json.load(f)
    return summary, histories

def plot_final_test_accuracy_histogram(exp_names, test_set):
    """
    Loads saved models, evaluates them on the test set, and plots a
    horizontal histogram of the calculated test accuracies.
    """
    if test_set is None:
        print("Skipping test accuracy histogram: Test data not loaded.")
        return
    test_imgs, test_labs = test_set
    
    print("\nCalculating final test accuracies by loading models...")
    accuracies = {}
    for exp_name in exp_names:
        model_path = os.path.join(MODELS_DIR, exp_name, 'best_model.pickle')
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found for {exp_name} at {model_path}. Skipping.")
            continue

        print(f"  Loading and evaluating: {exp_name}")
        # Create a new model instance with the correct architecture
        model = Model_MLP(
            size_list=MLP_SIZE_LIST,
            act_func=MLP_ACT_FUNC,
            lambda_list=MLP_LAMBDA_LIST
        )
        # Load the saved weights
        try:
            model.load_model(model_path) # Assumes model has load_model method
        except Exception as e:
            print(f"Error loading model {exp_name}: {e}. Skipping.")
            continue

        # Evaluate on the test set
        try:
            logits = model(test_imgs) # Forward pass
            acc = accuracy(logits, test_labs) # Calculate accuracy
            accuracies[exp_name] = acc
            print(f"    {exp_name} Test Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"Error evaluating model {exp_name}: {e}. Skipping.")
            continue

    if not accuracies:
        print("No models were successfully evaluated. Cannot plot histogram.")
        return

    # Prepare data for plotting
    plot_exp_names = list(accuracies.keys())
    plot_accuracies = [accuracies[name] for name in plot_exp_names]

    # Sort by accuracy for better visualization
    sorted_pairs = sorted(zip(plot_accuracies, plot_exp_names))
    accuracies_sorted, exp_names_sorted = zip(*sorted_pairs)

    # --- Plotting ---
    plt.figure(figsize=(8, 6)) # Adjusted figure size for histogram
    y_pos = np.arange(len(exp_names_sorted))
    bars = plt.barh(y_pos, accuracies_sorted, align='center')
    plt.yticks(y_pos, exp_names_sorted)
    plt.xlabel('Final Test Accuracy (Calculated)')
    plt.title('Comparison of Final Test Accuracy Across Experiments')
    # Adjust x-limits slightly beyond min/max accuracy
    min_acc = min(accuracies_sorted)
    max_acc = max(accuracies_sorted)
    plt.xlim(min_acc - (max_acc-min_acc)*0.05 - 0.001, max_acc + (max_acc-min_acc)*0.05 + 0.001)

    # Add accuracy values on bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.4f}', # Format to 4 decimal places
                 va='center', ha='left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'final_test_accuracy_histogram_calculated.png'), dpi=600) # Changed filename
    plt.close()
    print("Final test accuracy histogram saved.")

def plot_validation_loss_comparison(histories):
    """Plots validation loss over epochs for all experiments."""
    plt.figure(figsize=(10, 6))
    for i, (exp_name, history) in enumerate(histories.items()):
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            plt.plot(epochs, history['val_loss'], '-', marker=MARKERS[i % len(MARKERS)], markevery=5, label=exp_name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison of Different Settings')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'validation_loss_comparison.png'), dpi=600)
    plt.close()

def plot_theoretical_learning_rate_change(initial_lr=0.1, total_epochs=30):
    """Plots theoretical learning rate changes for different schedulers."""
    epochs = np.arange(1, total_epochs + 1)
    plt.figure(figsize=(10, 6))

    # --- StepLR ---
    step_size = 10
    gamma_step = 0.1
    lr_step = initial_lr * (gamma_step ** (np.floor((epochs - 1) / step_size)))
    plt.plot(epochs, lr_step, '-', marker=MARKERS[0], markevery=5, label=f'StepLR (step={step_size}, gamma={gamma_step})')

    # --- MultiStepLR ---
    milestones = [15, 25]
    gamma_multi = 0.1
    lr_multi = np.full_like(epochs, initial_lr, dtype=float)
    for i, epoch in enumerate(epochs):
        power = sum(1 for m in milestones if epoch > m)
        lr_multi[i] = initial_lr * (gamma_multi ** power)
    plt.plot(epochs, lr_multi, '-', marker=MARKERS[1], markevery=5, label=f'MultiStepLR (milestones={milestones}, gamma={gamma_multi})')

    # --- ExponentialLR ---
    gamma_exp = 0.95
    lr_exp = initial_lr * (gamma_exp ** (epochs - 1))
    plt.plot(epochs, lr_exp, '-', marker=MARKERS[2], markevery=5, label=f'ExponentialLR (gamma={gamma_exp})')

    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Theoretical Learning Rate Decay')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Keep log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'theoretical_learning_rate_change.png'), dpi=600)
    plt.close()

if __name__ == "__main__":
    # Load summary to get experiment names and histories for validation loss plot
    summary, histories = load_results()
    
    # Load test data once
    test_set = load_mnist_test_data()

    if summary: # Check if summary loaded successfully (needed for exp_names)
        exp_names = list(summary.keys())
        plot_final_test_accuracy_histogram(exp_names, test_set)
    else:
        print("Could not load summary.json, skipping final test accuracy histogram.")

    if histories: # Check if histories loaded successfully
        plot_validation_loss_comparison(histories)
    else:
        print("Could not load history files, skipping validation loss comparison.")

    # Plot theoretical LR change regardless of experiment results
    plot_theoretical_learning_rate_change()

    print("\nPlotting script finished.")