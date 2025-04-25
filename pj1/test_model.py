# Script to load a trained model and evaluate it on the test dataset.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt  # Imported but not used in this script
import pickle
import os  # Import os for path joining

# --- Model Loading ---
# Specify the path to the saved model file
# This should match the save_dir and filename used in test_train.py
saved_model_path = './saved_models/mlp_best/best_model.pickle'

print(f"Loading model from: {saved_model_path}")
model = nn.models.Model_MLP()  # Initialize an empty MLP model structure
# Load the trained parameters into the model
model.load_model(saved_model_path)
print("Model loaded successfully.")

# --- Test Data Loading ---
# Use standardized paths for the test dataset
data_dir = 'dataset/MNIST'
test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

print(f"Loading test images from: {test_images_path}")
with gzip.open(test_images_path, 'rb') as f:
    # Read the magic number, number of images, rows, and columns
    magic, num_images, rows, cols = unpack('>4I', f.read(16))
    print(
        f"Test Images - Magic: {magic}, Num: {num_images}, Rows: {rows}, Cols: {cols}")
    # Read the image data and reshape
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(
        num_images, rows * cols)

print(f"Loading test labels from: {test_labels_path}")
with gzip.open(test_labels_path, 'rb') as f:
    # Read the magic number and number of labels
    magic, num_labels = unpack('>2I', f.read(8))
    print(f"Test Labels - Magic: {magic}, Num: {num_labels}")
    assert num_images == num_labels, "Number of test images and labels must match."
    # Read the label data
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

print("Test data loaded successfully.")

# --- Data Preprocessing ---
# Normalize pixel values from [0, 255] to [0, 1]
# Divide by 255.0 to handle potential max value of 0 in a batch (though unlikely for MNIST test set)
test_imgs = test_imgs / 255.0
print("Test data normalized.")

# --- Evaluation ---
print("Evaluating model on test data...")
logits = model(test_imgs)  # Perform forward pass on the test data
accuracy = nn.metric.accuracy(logits, test_labs)  # Calculate accuracy

# --- Output Result ---
print(f"\nTest Accuracy: {accuracy:.4f}")
