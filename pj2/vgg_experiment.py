import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
from vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Light, VGG_A_Dropout
from utils import plot_images, plot_loss_accuracy, plot_filters, plot_confusion_matrix, FIGURE_DIR, MODEL_DIR, load_cifar10_data, plot_loss_landscape
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

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

# Load CIFAR-10 data
trainloader, testloader = load_cifar10_data(transform_train, transform_test)

# Define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Visualize some training images
if len(trainloader) > 0:
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    plot_images(images, labels, classes)


def train_model(model, criterion, optimizer, num_epochs=25, patience=6):
    start_time = time.time()
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_acc = 0.0
    epochs_since_improvement = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(trainloader, desc=f'Training Epoch {epoch+1}')
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_loss = running_loss / (i + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.3f}',
                             'acc': f'{current_acc:.2f}%'})
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(testloader, desc=f'Validation Epoch {epoch+1}')
        with torch.no_grad():
            for data in pbar:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                current_loss = val_loss / (pbar.n + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix(
                    {'loss': f'{current_loss:.3f}', 'acc': f'{current_acc:.2f}%'})
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        avg_val_loss = val_loss / len(testloader)
        test_losses.append(avg_val_loss)
        print(f'Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            torch.save(model.state_dict(), os.path.join(
                MODEL_DIR, f'best_{model.__class__.__name__}.pth'))
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if epochs_since_improvement >= patience:
            print(
                f"Early stopping triggered after {epoch+1} epochs. No improvement in test accuracy for {patience} epochs.")
            break
    time_elapsed = time.time() - start_time
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    return model, train_losses, test_losses, train_accuracies, test_accuracies, best_acc


def test_model(model):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    for i in range(10):
        print(
            f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    plot_confusion_matrix(all_labels, all_preds, classes,
                          model.__class__.__name__)


def main():
    models = {
        "VGG_A": VGG_A(),
        "VGG_A_BatchNorm": VGG_A_BatchNorm(),
        "VGG_A_Light": VGG_A_Light(),
        "VGG_A_Dropout": VGG_A_Dropout()
    }
    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Training {name}")
        print(f"{'=' * 50}")
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model, train_losses, test_losses, train_accuracies, test_accuracies, best_acc = train_model(
            model, criterion, optimizer, num_epochs=60)
        results[name] = {
            "model": model,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
            "best_acc": best_acc,
            "params": total_params
        }
        test_model(model)
        plot_filters(model)
        if name in ["VGG_A", "VGG_A_BatchNorm"]:
            plot_loss_landscape(model, criterion, testloader, device)
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        plt.plot(result["test_accuracies"],
                 label=f"{name} (Best: {result['best_acc']:.2f}%, Params: {result['params']:,})")
    plt.title("VGG Model Comparison - Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIGURE_DIR, "vgg_model_comparison.png"))


if __name__ == "__main__":
    main()
