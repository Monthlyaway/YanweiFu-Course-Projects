#!/usr/bin/env python
# Weight visualization script for neural network models
import os
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import argparse # Keep for save_dir argument

# load_mnist_data function removed as it's not needed for visualizing pre-trained weights

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
            plt.close(fig) # Close the specific figure
        else:
            plt.show()
    
    # Visualize weight matrices as heatmaps for all linear layers
    for i, (idx, layer) in enumerate(linear_layers):
        weights = layer.params['W']
        
        fig_heatmap = plt.figure(figsize=(10, 8)) # Create a new figure
        plt.title(f"Layer {idx} Weight Matrix ({weights.shape[0]}x{weights.shape[1]})")
        im_heatmap = plt.imshow(weights, cmap='coolwarm', aspect='auto') # Use aspect='auto' for better visualization
        plt.colorbar(im_heatmap)
        plt.xlabel("Output Neurons")
        plt.ylabel("Input Neurons")
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'layer_{idx}_weights_heatmap.png'))
            plt.close(fig_heatmap) # Close the specific figure
        else:
            plt.show()
            
        # Show a histogram of weight distributions for all linear layers
        fig_hist = plt.figure(figsize=(8, 6)) # Create a new figure
        plt.hist(weights.flatten(), bins=50, alpha=0.7)
        plt.title(f"Layer {idx} Weight Distribution")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        
        if save_dir:
            # No need to create dir again if already created
            plt.savefig(os.path.join(save_dir, f'layer_{idx}_weight_hist.png'))
            plt.close(fig_hist) # Close the specific figure
        else:
            plt.show()

# visualize_cnn_weights function removed

def main():
    # Parse command line arguments - Keep only necessary ones
    parser = argparse.ArgumentParser(description='Visualize MLP weights for MomentGD_StepLR model')
    # parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file') # Hardcoded below
    # parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn'], default='mlp', help='Type of model (mlp or cnn)') # Only MLP now
    # parser.add_argument('--dataset_dir', type=str, default='./dataset/MNIST', help='Directory containing MNIST dataset') # Not needed for visualization
    parser.add_argument('--save_dir', type=str, default='visualization_results/mlp/MomentGD_StepLR', 
                        help='Directory to save visualizations (default: visualization_results/mlp/MomentGD_StepLR)')
    
    args = parser.parse_args()

    # Define the path to the specific model we want to visualize
    model_path = 'saved_models/mlp/MomentGD_StepLR/best_model.pickle'
    save_dir = args.save_dir # Use command line arg for save dir, with a default

    # Load the model
    try:
        model = nn.models.Model_MLP() # Only MLP model
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please ensure you have run the 'run_mlp_experiments.py' script first and that the model was saved correctly.")
            return
            
        model.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
         print(f"Error: Model file not found at {model_path}")
         print("Please ensure you have run the 'run_mlp_experiments.py' script first and that the model was saved correctly.")
         return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Visualize weights
    visualize_mlp_weights(model, save_dir) # Only visualize MLP weights
    
    if save_dir:
        # Ensure the save directory exists (it's also created in visualize_mlp_weights, but good practice)
        os.makedirs(save_dir, exist_ok=True) 
        print(f"Visualizations saved to {save_dir}")
    else:
        print("Displaying visualizations...")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()