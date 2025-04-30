import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

plt.style.use(['science'])

RESULTS_DIR = 'experiment_results/mlp'
FIGS_DIR = 'figs/mlp'
os.makedirs(FIGS_DIR, exist_ok=True)

MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']

# Load experiment results
def load_results():
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

# Plot learning rate schedules
def plot_learning_rates():
    init_lr = 0.01
    epochs = 20
    x = np.arange(1, epochs+1)
    step_lr = np.ones(epochs) * init_lr
    for i in range(epochs):
        if (i+1) % 5 == 0:  # Updated to match new step_size
            step_lr[i:] *= 0.9  # Updated to match new gamma
    multistep_lr = np.ones(epochs) * init_lr
    milestones_epochs = [1600//782, 4000//782, 10000//782]
    for i in range(epochs):
        if (i+1) in milestones_epochs:
            multistep_lr[i:] *= 0.5
    exp_lr = np.array([init_lr * (0.995 ** i) for i in range(epochs)])
    plt.figure(figsize=(10, 6))
    plt.plot(x, step_lr, '-', label='StepLR (gamma=0.9, every 5 epochs)')  # Updated label
    plt.plot(x, multistep_lr, '-', label='MultiStepLR (gamma=0.5, milestones)')
    plt.plot(x, exp_lr, '-', label='ExponentialLR (gamma=0.995)')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedulers Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'learning_rate_comparison.png'), dpi=600)
    plt.close()

# Plot optimizer comparison
def plot_optimizer_comparison(histories):
    optimizers = ['SGD', 'MomentGD']
    for i, opt in enumerate(optimizers):
        exp_name = f"{opt}_StepLR"  # Use StepLR for optimizer comparison
        if exp_name in histories:
            history = histories[exp_name]
            epochs = range(1, len(history['val_accuracy']) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['val_accuracy'], '-', marker=MARKERS[i], markevery=5, label=opt)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Optimizer Comparison (StepLR)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'optimizer_comparison.png'), dpi=600)
    plt.close()

# Plot scheduler comparison
def plot_scheduler_comparison(histories):
    schedulers = ['StepLR', 'MultiStepLR', 'ExponentialLR']
    scheduler_labels = ['StepLR', 'MultiStepLR', 'ExponentialLR']
    for i, (sch, label) in enumerate(zip(schedulers, scheduler_labels)):
        exp_name = f"MomentGD_{sch}"
        if exp_name in histories:
            history = histories[exp_name]
            epochs = range(1, len(history['val_accuracy']) + 1)
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['val_accuracy'], '-', marker=MARKERS[i], markevery=5, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Scheduler Comparison (MomentGD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'scheduler_comparison.png'), dpi=600)
    plt.close()

# Plot best validation accuracy for all experiments
def plot_best_accuracy_comparison(summary):
    if not summary:
        return
    exp_names = list(summary.keys())
    best_accuracies = [summary[name]['best_val_accuracy'] for name in exp_names]
    sorted_indices = np.argsort(best_accuracies)[::-1]
    sorted_names = [exp_names[i] for i in sorted_indices]
    sorted_accuracies = [best_accuracies[i] for i in sorted_indices]
    bars = plt.barh(range(len(sorted_names)), sorted_accuracies)
    for i, v in enumerate(sorted_accuracies):
        plt.text(v + 0.001, i, f"{v:.4f}", va='center')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Best Validation Accuracy')
    plt.title('Best Validation Accuracy of All Experiments')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'best_accuracy_comparison.png'), dpi=600)
    plt.close()

def main():
    print("Plotting MLP experiment results...")
    plot_learning_rates()
    print("Learning rate schedule plot saved.")
    summary, histories = load_results()
    if not histories:
        print("Experiment results not found. Please run experiments first.")
        return
    plot_optimizer_comparison(histories)
    print("Optimizer comparison plot saved.")
    plot_scheduler_comparison(histories)
    print("Scheduler comparison plot saved.")
    plot_best_accuracy_comparison(summary)
    print("Best accuracy comparison plot saved.")
    print("All plots saved to", FIGS_DIR)

if __name__ == "__main__":
    main()