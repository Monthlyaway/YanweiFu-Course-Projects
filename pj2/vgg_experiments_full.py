from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
from vgg import VGG_A, VGG_A_BatchNorm
from utils import load_cifar10_data, FIGURE_DIR, MODEL_DIR
import matplotlib as mpl
mpl.use('Agg')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def set_random_seeds(seed_value=0, device='cpu'):
    import random
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# Hyperparameters
num_epochs = 200
batch_size = 128
subset_size = None  # Set to e.g. 2000 for quick runs
# Multiple learning rates for loss landscape
learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

# Load data
trainloader, testloader = load_cifar10_data(
    transform_train, transform_test, batch_size_train=batch_size, batch_size_test=100, subset_size=subset_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def train_model_with_step_losses(model, criterion, optimizer, trainloader, testloader, num_epochs=20, patience=10):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    step_losses = []  # Record loss at every batch (step)
    best_acc = 0.0
    epochs_since_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step_losses.append(loss.item())  # Record step loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(100 * correct / total)
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_losses.append(val_loss / len(testloader))
        test_accuracies.append(100 * correct / total)
        if test_accuracies[-1] > best_acc:
            best_acc = test_accuracies[-1]
            torch.save(model.state_dict(), os.path.join(
                MODEL_DIR, f'best_{model.__class__.__name__}.pth'))
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        print(
            f"Epoch {epoch+1}: Train Acc {train_accuracies[-1]:.2f}%, Test Acc {test_accuracies[-1]:.2f}%")
    return train_losses, test_losses, train_accuracies, test_accuracies, step_losses


def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{model_name}_training_curves.png'))
    plt.close()


def plot_loss_landscape():
    """Load saved step losses and plot loss landscape for both models"""
    # Load all saved step losses
    models_names = ['VGG_A', 'VGG_A_BatchNorm']
    model_landscapes = {}

    for model_name in models_names:
        all_step_losses = []
        for lr in learning_rates:
            filename = os.path.join(
                MODEL_DIR, f'{model_name}_lr_{lr}_step_losses.npy')
            if os.path.exists(filename):
                step_losses = np.load(filename)
                all_step_losses.append(step_losses)
            else:
                print(f"Warning: {filename} not found")

        if all_step_losses:
            # Align lengths - use minimum length across all learning rates
            min_len = min(len(losses) for losses in all_step_losses)
            all_step_losses = [losses[:min_len] for losses in all_step_losses]

            # Convert to numpy array for easier computation
            all_step_losses = np.array(all_step_losses)

            # Compute min and max curves
            min_curve = np.min(all_step_losses, axis=0)
            max_curve = np.max(all_step_losses, axis=0)

            model_landscapes[model_name] = {
                'min_curve': min_curve,
                'max_curve': max_curve
            }

    # Plot loss landscape
    plt.figure(figsize=(10, 6))

    # Plot VGG_A (Standard VGG) in green
    if 'VGG_A' in model_landscapes:
        steps = np.arange(len(model_landscapes['VGG_A']['min_curve']))
        plt.fill_between(steps,
                         model_landscapes['VGG_A']['min_curve'],
                         model_landscapes['VGG_A']['max_curve'],
                         color='green', alpha=0.3, label='Standard VGG')

    # Plot VGG_A_BatchNorm in red
    if 'VGG_A_BatchNorm' in model_landscapes:
        steps = np.arange(
            len(model_landscapes['VGG_A_BatchNorm']['min_curve']))
        plt.fill_between(steps,
                         model_landscapes['VGG_A_BatchNorm']['min_curve'],
                         model_landscapes['VGG_A_BatchNorm']['max_curve'],
                         color='red', alpha=0.3, label='Standard VGG + BatchNorm')

    plt.title('Loss Landscape (per training step)')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURE_DIR, 'vggA_vs_vggA_BN_loss_landscape_steps.png'))
    plt.close()
    print("Loss landscape plot saved")


def main():
    set_random_seeds(2020, device)
    models = {
        'VGG_A': VGG_A,
        'VGG_A_BatchNorm': VGG_A_BatchNorm,
    }
    criterion = nn.CrossEntropyLoss()

    # Train each model with different learning rates
    for name, model_class in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name} with different learning rates")
        print(f"{'='*50}")

        for idx, lr in enumerate(learning_rates):
            print(f"\nTraining {name} with learning rate: {lr}")

            # Create new model instance for each learning rate
            model = model_class().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train model
            train_losses, test_losses, train_accuracies, test_accuracies, step_losses = train_model_with_step_losses(
                model, criterion, optimizer, trainloader, testloader, num_epochs=num_epochs)

            # Save step losses to numpy file
            step_losses_file = os.path.join(
                MODEL_DIR, f'{name}_lr_{lr}_step_losses.npy')
            np.save(step_losses_file, np.array(step_losses))
            print(f"Saved step losses to {step_losses_file}")

            # Only plot curves for the first learning rate
            if idx == 0:
                plot_curves(train_losses, test_losses,
                            train_accuracies, test_accuracies, name)

    # Plot loss landscape after all models are trained
    print("\nPlotting loss landscape...")
    plot_loss_landscape()

    print('\nAll figures saved in', FIGURE_DIR)
    print('All model step losses saved in', MODEL_DIR)


if __name__ == "__main__":
    main()
