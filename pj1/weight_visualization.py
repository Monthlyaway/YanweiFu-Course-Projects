#!/usr/bin/env python
# Weight visualization script for neural network models
import os
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import argparse

def load_mnist_data(dataset_dir='./dataset/MNIST'):
    """
    Load MNIST dataset from the specified directory.
    
    Args:
        dataset_dir: Directory containing the MNIST dataset files
        
    Returns:
        test_imgs: Normalized test images
        test_labs: Test labels
    """
    test_images_path = os.path.join(dataset_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(dataset_dir, 't10k-labels-idx1-ubyte.gz')

    # Check if files exist
    if not os.path.exists(test_images_path) or not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"MNIST dataset files not found in {dataset_dir}")

    # Load test images
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
    # Load test labels
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    # Normalize images to [0, 1] range
    test_imgs = test_imgs / 255.0

    return test_imgs, test_labs

def visualize_mlp_weights(model, save_dir=None):
    """
    Visualize weights of an MLP model.
    
    Args:
        model: The MLP model to visualize
        save_dir: Directory to save visualizations. If None, plots are displayed.
    """
    # Check if model has layers
    if not hasattr(model, 'layers') or len(model.layers) == 0:
        print("Model has no layers to visualize")
        return
    
    # Find all Linear layers
    linear_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.op.Linear):
            linear_layers.append((i, layer))
    
    if not linear_layers:
        print("No Linear layers found in the model")
        return
    
    # Visualize first layer weights (input to hidden)
    first_layer = linear_layers[0][1]
    first_weights = first_layer.params['W']
    
    # For the first layer in an MNIST MLP, weights can be visualized as 28x28 images
    if first_weights.shape[0] == 784:  # MNIST input dimension
        fig, axes = plt.subplots(5, 8, figsize=(15, 10))
        fig.suptitle('First Layer Weights (Input Features)', fontsize=16)
        fig.tight_layout(pad=3.0)
        
        axes = axes.flatten()
        # Select a subset of neurons to display (up to 40)
        num_to_show = min(40, first_weights.shape[1])
        
        for i in range(num_to_show):
            ax = axes[i]
            # Reshape weights to 28x28 for visualization
            weight_img = first_weights[:, i].reshape(28, 28)
            im = ax.imshow(weight_img, cmap='viridis')
            ax.set_title(f"Neuron {i}")
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(num_to_show, len(axes)):
            axes[i].axis('off')
            
        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'first_layer_weights.png'))
            plt.close()
        else:
            plt.show()
    
    # Visualize weight matrices as heatmaps
    for i, (idx, layer) in enumerate(linear_layers):
        weights = layer.params['W']
        
        plt.figure(figsize=(10, 8))
        plt.title(f"Layer {idx} Weight Matrix ({weights.shape[0]}x{weights.shape[1]})")
        im = plt.imshow(weights, cmap='coolwarm')
        plt.colorbar(im)
        plt.xlabel("Output Neurons")
        plt.ylabel("Input Neurons")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'layer_{idx}_weights.png'))
            plt.close()
        else:
            plt.show()
            
        # If it's not the last layer, show a histogram of weight distributions
        if i < len(linear_layers) - 1:
            plt.figure(figsize=(8, 6))
            plt.hist(weights.flatten(), bins=50, alpha=0.7)
            plt.title(f"Layer {idx} Weight Distribution")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'layer_{idx}_weight_hist.png'))
                plt.close()
            else:
                plt.show()

def visualize_cnn_weights(model, save_dir=None):
    """
    Visualize weights of a CNN model.
    
    Args:
        model: The CNN model to visualize
        save_dir: Directory to save visualizations. If None, plots are displayed.
    """
    # Check if model has layers
    if not hasattr(model, 'layers') or len(model.layers) == 0:
        print("Model has no layers to visualize")
        return
    
    # Find all conv2D layers
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.op.conv2D):
            conv_layers.append((i, layer))
    
    if not conv_layers:
        print("No conv2D layers found in the model")
        return
    
    # Visualize convolutional filters
    for layer_idx, (idx, layer) in enumerate(conv_layers):
        filters = layer.params['W']  # Shape: [out_channels, in_channels, kernel_height, kernel_width]
        
        out_channels, in_channels, k_h, k_w = filters.shape
        
        # Create a grid to display filters
        grid_size = int(np.ceil(np.sqrt(out_channels)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'Layer {idx} Convolutional Filters', fontsize=16)
        fig.tight_layout(pad=3.0)
        
        # Make axes indexable for any grid size
        if grid_size == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # For each output channel, display the filters for each input channel
        for i in range(out_channels):
            if i < len(axes):
                # For filters with multiple input channels, average them for display
                if in_channels > 1:
                    # Average across input channels
                    filter_img = np.mean(filters[i], axis=0)
                else:
                    filter_img = filters[i, 0]
                
                # Display the filter
                im = axes[i].imshow(filter_img, cmap='viridis')
                axes[i].set_title(f"Filter {i}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        # Hide unused subplots
        for i in range(out_channels, len(axes)):
            axes[i].axis('off')
        
        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'conv_layer_{idx}_filters.png'))
            plt.close()
        else:
            plt.show()
        
        # If this isn't the first layer, we can also look at the first filter in more detail
        if layer_idx > 0 and in_channels > 1:
            # Show each input channel for the first filter
            fig, axes = plt.subplots(1, in_channels, figsize=(in_channels * 3, 3))
            fig.suptitle(f'Layer {idx}, Filter 0 - All Input Channels', fontsize=14)
            
            if in_channels == 1:
                axes = [axes]
                
            for c in range(in_channels):
                im = axes[c].imshow(filters[0, c], cmap='viridis')
                axes[c].set_title(f"Channel {c}")
                axes[c].set_xticks([])
                axes[c].set_yticks([])
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'conv_layer_{idx}_filter0_channels.png'))
                plt.close()
            else:
                plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize neural network weights')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn'], default='mlp', help='Type of model (mlp or cnn)')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/MNIST', help='Directory containing MNIST dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save visualizations (optional)')
    
    args = parser.parse_args()
    
    # Load the model
    try:
        if args.model_type == 'mlp':
            model = nn.models.Model_MLP()
        else:  # cnn
            model = nn.models.Model_CNN()
            
        model.load_model(args.model_path)
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Visualize weights
    if args.model_type == 'mlp':
        visualize_mlp_weights(model, args.save_dir)
    else:  # cnn
        visualize_cnn_weights(model, args.save_dir)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()