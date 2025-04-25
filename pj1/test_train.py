# An example of read in the data and train the model.
# The runner is implemented, while the model used for training needs your implementation.
import mynn as nn
from draw_tools.plot import plot  # Assumes this utility exists and works

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os  # Import os for path joining if needed, though forward slashes work generally

# Fixed seed for reproducible experiments
np.random.seed(309)

# --- Data Loading ---
# Use forward slashes for better cross-platform compatibility
data_dir = 'dataset/MNIST'
train_images_path = os.path.join(
    data_dir, 'train-images-idx3-ubyte.gz')  # Example using os.path.join
train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')

print(f"Loading training images from: {train_images_path}")
with gzip.open(train_images_path, 'rb') as f:
    # Read the magic number, number of images, rows, and columns
    magic, num_images, rows, cols = unpack('>4I', f.read(16))
    print(
        f"Images - Magic: {magic}, Num: {num_images}, Rows: {rows}, Cols: {cols}")
    # Read the image data and reshape
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(
        num_images, rows * cols)

print(f"Loading training labels from: {train_labels_path}")
with gzip.open(train_labels_path, 'rb') as f:
    # Read the magic number and number of labels
    magic, num_labels = unpack('>2I', f.read(8))
    print(f"Labels - Magic: {magic}, Num: {num_labels}")
    assert num_images == num_labels, "Number of images and labels must match."
    # Read the label data
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

print("Data loaded successfully.")

# --- Data Preprocessing ---
# Shuffle dataset
idx = np.random.permutation(np.arange(num_images))
# Optional: save the index for reproducibility if needed elsewhere
# with open('idx.pickle', 'wb') as f:
#         pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]

# Split into training and validation sets
num_validation = 10000
valid_imgs = train_imgs[:num_validation]
valid_labs = train_labs[:num_validation]
train_imgs = train_imgs[num_validation:]
train_labs = train_labs[num_validation:]
print(
    f"Training samples: {train_imgs.shape[0]}, Validation samples: {valid_imgs.shape[0]}")

# Normalize pixel values from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.0
valid_imgs = valid_imgs / 255.0
print("Data normalized.")

# --- Model, Optimizer, Loss, Scheduler Setup ---
# Define MLP model: Input(784) -> Linear(600) -> ReLU -> Linear(10) -> Softmax (in Loss)
# Weight decay lambda applied to each linear layer
linear_model = nn.models.Model_MLP(
    # [Input_dim, Hidden_dim, Output_dim]
    size_list=[train_imgs.shape[-1], 600, 10],
    act_func='ReLU',
    lambda_list=[1e-4, 1e-4]  # Weight decay for layer 1 and layer 2
)
print(f"Model: {linear_model.__class__.__name__}")

# Define Optimizer
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
optimizer = nn.optimizer.MomentGD(init_lr=0.01, model=linear_model, mu=0.9) # Example using MomentGD
# optimizer = nn.optimizer.AlternativeMomentGD(init_lr=0.1, model=linear_model, beta=0.0005) # Using AlternativeMomentGD
print(f"Optimizer: {optimizer.__class__.__name__}")

# Define Learning Rate Scheduler
# Note: RunnerM steps the scheduler per *iteration* by default.
# These milestones refer to total iteration counts across epochs.
scheduler = nn.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=[800, 2400, 4000],  # Iteration milestones for LR decay
    gamma=0.5
)
print(f"Scheduler: {scheduler.__class__.__name__}")

# Define Loss Function (includes Softmax)
# loss_fn = nn.op.MultiCrossEntropyLoss(
#     model=linear_model,
#     max_classes=int(train_labs.max() + 1)  # Should be 10 for MNIST
# )
loss_fn = nn.op.CrossEntropyLoss(
    model=linear_model,
    num_classes=int(train_labs.max() + 1)  # Should be 10 for MNIST
)
print(f"Loss function: {loss_fn.__class__.__name__}")

# --- Runner Setup ---
# Use forward slash for save_dir path
save_directory = './saved_models/mlp_best'  # Changed save directory slightly
runner = nn.runner.RunnerM(
    model=linear_model,
    optimizer=optimizer,
    metric=nn.metric.accuracy,
    loss_fn=loss_fn,
    scheduler=scheduler,
    batch_size=64  # Example: Set batch size in runner
)
print("Runner created.")

# --- Training ---
runner.train(
    train_set=[train_imgs, train_labs],
    dev_set=[valid_imgs, valid_labs],
    num_epochs=5,
    log_iters=100,  # Log progress every 100 iterations
    save_dir=save_directory
)

# --- Plotting Results ---
print("Plotting training history...")
# Create figure and axes for plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes = axes.reshape(-1) # Not needed if axes is already 1D or 2D as expected by plot
fig.suptitle("MLP Training History")  # Add a title to the figure
plot(runner, axes)  # Pass runner and axes to the plotting function
# Adjust layout to prevent title overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("Script finished.")
