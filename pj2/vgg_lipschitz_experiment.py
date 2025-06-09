import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
from vgg import VGG_A, VGG_A_BatchNorm
from utils import load_cifar10_data, FIGURE_DIR
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

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

# Load CIFAR-10 data (use testloader for evaluation)
_, testloader = load_cifar10_data(transform_train, transform_test)

def compute_lipschitz_curve(model, criterion, dataloader, device, lr_list, steps=1):
    model.eval()
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    # Compute original loss and gradient
    outputs = model(images)
    loss = criterion(outputs, labels)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    # Flatten gradients
    grad_vector = torch.cat([g.detach().view(-1) for g in grads])
    grad_norm = torch.norm(grad_vector)
    grad_unit = grad_vector / (grad_norm + 1e-12)
    # Store original parameters
    orig_params = [p.detach().clone() for p in model.parameters()]
    loss_curves = []
    for lr in lr_list:
        # Step in negative gradient direction
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            direction = grad_unit[idx:idx+numel].view_as(p)
            p.data = p.data - lr * direction
            idx += numel
        # Compute new loss
        outputs_new = model(images)
        loss_new = criterion(outputs_new, labels).item()
        loss_curves.append(loss_new)
        # Restore parameters
        for p, orig in zip(model.parameters(), orig_params):
            p.data = orig.data.clone()
    return loss_curves

def main():
    criterion = nn.CrossEntropyLoss()
    lr_list = [1e-3, 2e-3, 1e-4, 5e-4]
    models = {
        'VGG_A': VGG_A(),
        'VGG_A_BatchNorm': VGG_A_BatchNorm()
    }
    results = {}
    for name, model in models.items():
        print(f"\nComputing Lipschitz curve for {name}")
        model = model.to(device)
        # Optionally load pretrained weights if available
        model_path = os.path.join('models', f'best_{name}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"No pretrained weights found for {name}, using random init.")
        loss_curve = compute_lipschitz_curve(model, criterion, testloader, device, lr_list)
        results[name] = loss_curve
    # For each learning rate, get max/min across models
    max_curve = np.maximum(results['VGG_A'], results['VGG_A_BatchNorm'])
    min_curve = np.minimum(results['VGG_A'], results['VGG_A_BatchNorm'])
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(lr_list, results['VGG_A'], 'o-', label='VGG_A (no BN)')
    plt.plot(lr_list, results['VGG_A_BatchNorm'], 's-', label='VGG_A_BatchNorm')
    plt.fill_between(lr_list, min_curve, max_curve, color='gray', alpha=0.3, label='Gap (max-min)')
    plt.xscale('log')
    plt.xlabel('Learning Rate (step size)')
    plt.ylabel('Loss after step in -grad direction')
    plt.title('Lipschitzness of Loss: VGG_A vs VGG_A_BatchNorm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'vgg_loss_lipschitz_comparison.png'))
    print(f"Saved plot to {os.path.join(FIGURE_DIR, 'vgg_loss_lipschitz_comparison.png')}")

if __name__ == "__main__":
    main() 