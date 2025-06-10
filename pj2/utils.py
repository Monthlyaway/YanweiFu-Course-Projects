import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchviz import make_dot
import os
import kagglehub
import torchvision

FIGURE_DIR = "figures"
MODEL_DIR = "models"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def plot_images(images, labels, classes):
    """
    Plot a batch of images with their labels.
    """
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i + 1)
        # Convert tensor to numpy array and transpose from CxHxW to HxWxC
        img = images[i].numpy().transpose((1, 2, 0))
        # Unnormalize the image
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'sample_images.png'))


def plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies, model_name):
    """
    Plot training and validation loss and accuracy.
    """
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
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{model_name}_training_curves.png'))


def plot_filters(model):
    """
    Visualize filters from the first convolutional layer.
    """
    # Get the first convolutional layer
    first_conv_layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv_layer = module
            break

    if first_conv_layer is None:
        print("No convolutional layer found in the model.")
        return

    # Get the weights
    weights = first_conv_layer.weight.data.cpu()

    # Normalize the weights for better visualization
    min_val = weights.min()
    max_val = weights.max()
    weights = (weights - min_val) / (max_val - min_val)

    # Plot the filters
    num_filters = min(64, weights.shape[0])
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            if idx < num_filters:
                # For RGB filters, take the average across channels
                filter_img = weights[idx].mean(dim=0)
                axes[i, j].imshow(filter_img, cmap='viridis')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURE_DIR, f'{model.__class__.__name__}_filters.png'), dpi=600)


def plot_confusion_matrix(y_true, y_pred, classes, model_name="model"):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURE_DIR, f'confusion_matrix_{model_name}.png'), dpi=600)


def visualize_model_architecture(model, input_size=(3, 32, 32)):
    """
    Visualize the model architecture using torchviz.
    """
    x = torch.randn(1, *input_size).requires_grad_(True)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render(os.path.join(
        FIGURE_DIR, f'{model.__class__.__name__}_architecture'), format='png', dpi=600)


def compute_flops(model, input_size=(3, 32, 32)):
    """
    Estimate the number of FLOPs for a model.
    """
    def count_conv2d(m, x, y):
        x = x[0]
        batch_size = x.size(0)
        output_height, output_width = y.size(2), y.size(3)

        kernel_height, kernel_width = m.kernel_size
        in_channels = m.in_channels
        out_channels = m.out_channels

        # Number of operations per element
        ops_per_element = kernel_height * kernel_width * in_channels

        # Total number of output elements
        output_elements = batch_size * out_channels * output_height * output_width

        # Total FLOPs
        total_ops = output_elements * ops_per_element

        m.total_ops = torch.DoubleTensor([int(total_ops)])

    def count_linear(m, x, y):
        x = x[0]
        batch_size = x.size(0) if len(x.size()) > 1 else 1
        ops_per_element = m.in_features
        total_ops = batch_size * m.out_features * ops_per_element
        m.total_ops = torch.DoubleTensor([int(total_ops)])

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv2d))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))

    # Run a forward pass
    x = torch.randn(1, *input_size)
    model(x)

    # Sum up the total operations
    total_ops = 0
    for name, module in model.named_modules():
        if hasattr(module, 'total_ops'):
            total_ops += module.total_ops.item()

    # Remove hooks
    for h in hooks:
        h.remove()

    return total_ops


def plot_loss_landscape(model, criterion, dataloader, device):
    """
    Plot the loss landscape by perturbing the model parameters.
    """
    # Get a batch of data
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # Compute the original loss
    original_params = [p.clone().detach() for p in model.parameters()]
    outputs = model(images)
    original_loss = criterion(outputs, labels).item()

    # Define a grid of perturbations
    alpha_range = np.linspace(-1, 1, 20)
    beta_range = np.linspace(-1, 1, 20)
    loss_landscape = np.zeros(
        (len(beta_range), len(alpha_range)))  # Fixed shape

    # Get two random directions
    direction1 = [torch.randn_like(p) for p in original_params]
    direction2 = [torch.randn_like(p) for p in original_params]

    # Normalize directions
    norm1 = torch.sqrt(sum(torch.sum(d * d) for d in direction1))
    norm2 = torch.sqrt(sum(torch.sum(d * d) for d in direction2))

    direction1 = [d / norm1 for d in direction1]
    direction2 = [d / norm2 for d in direction2]

    # Compute loss landscape
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Update model parameters
            for p, p0, d1, d2 in zip(model.parameters(), original_params, direction1, direction2):
                p.data = p0 + alpha * d1 + beta * d2

            # Compute loss
            outputs = model(images)
            loss = criterion(outputs, labels).item()
            loss_landscape[j, i] = loss  # Fixed indexing

    # Restore original parameters
    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0

    # Plot the loss landscape
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(alpha_range, beta_range)
    plt.contourf(X, Y, loss_landscape, 20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.title('Loss Landscape')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')

    # Add a marker for the original point
    plt.plot(0, 0, 'r*', markersize=15, label='Current parameters')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(
        FIGURE_DIR, f'{model.__class__.__name__}_loss_landscape.png'), dpi=600)

    # Optional: Also create a 3D visualization
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, loss_landscape, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('3D Loss Landscape')
    fig.colorbar(surf)
    plt.savefig(os.path.join(
        FIGURE_DIR, f'{model.__class__.__name__}_loss_landscape_3d.png'), dpi=600)


def get_accuracy(model, dataloader, device):
    """Compute accuracy of a model on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def set_random_seeds(seed_value=0, device='cpu'):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_cifar10_data(transform_train, transform_test, batch_size_train=128, batch_size_test=100, subset_size=None):
    # Download latest version
    path = kagglehub.dataset_download(
        "harshajakkam/cifar-10-python-cifar-10-python-tar-gz")
    print("Path to dataset files:", path)

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train)
    if subset_size is not None:
        trainset = torch.utils.data.Subset(trainset, range(subset_size))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test)
    if subset_size is not None:
        testset = torch.utils.data.Subset(testset, range(subset_size))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=8)

    return trainloader, testloader
