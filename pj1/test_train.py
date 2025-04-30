import sys  # 导入sys模块用于处理命令行参数
import os  # Import os for path joining if needed, though forward slashes work generally
import pickle
import matplotlib.pyplot as plt
import gzip
from struct import unpack
import numpy as np
from draw_tools.plot import plot  # Assumes this utility exists and works
import mynn as nn
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 命令行程序用于训练MNIST数据集上的MLP或CNN模型
# 用法: python test_train.py [mlp|cnn]


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

# --- 检查命令行参数 ---
if len(sys.argv) != 2 or sys.argv[1].lower() not in ['mlp', 'cnn']:
    print("用法: python test_train.py [mlp|cnn]")
    print("请指定模型类型: mlp 或 cnn")
    sys.exit(1)

model_type = sys.argv[1].lower()
print(f"选择的模型类型: {model_type}")

# --- Model, Optimizer, Loss, Scheduler Setup ---
if model_type == 'mlp':
    # 定义MLP模型: Input(784) -> Linear(600) -> ReLU -> Linear(10) -> Softmax (in Loss)
    model = nn.models.Model_MLP(
        # [Input_dim, Hidden_dim, Output_dim]
        size_list=[train_imgs.shape[-1], 600, 10],
        act_func='ReLU',
        lambda_list=[1e-4, 1e-4]  # Weight decay for layer 1 and layer 2
    )
    save_directory = './saved_models/mlp_best'
    init_lr = 0.01
    mu = 0.9
    milestones = [800, 2400, 4000]
    batch_size = 64
else:  # CNN模型
    # 对于CNN，需要将数据重塑为4D张量 [batch_size, channels, height, width]
    # 重塑训练数据
    train_imgs_reshaped = train_imgs.reshape(train_imgs.shape[0], 1, 28, 28)
    valid_imgs_reshaped = valid_imgs.reshape(valid_imgs.shape[0], 1, 28, 28)

    # 使用重塑后的数据
    train_imgs = train_imgs_reshaped
    valid_imgs = valid_imgs_reshaped

    # 定义CNN模型
    model = nn.models.Model_CNN(
        in_channels=1,  # MNIST是灰度图像，只有1个通道
        # [conv1_filters, conv2_filters, fc_hidden, output]
        filter_sizes=[2, 3, 32, 10],
        act_func='ReLU',
        lambda_list=[1e-4, 1e-4, 1e-4, 1e-4]  # 每层的权重衰减
    )
    save_directory = './saved_models/cnn_best'
    init_lr = 0.005
    mu = 0.9
    milestones = [800, 2400, 4000]
    batch_size = 32

print(f"Model: {model.__class__.__name__}")

# 定义优化器
optimizer = nn.optimizer.MomentGD(init_lr=init_lr, model=model, mu=mu)
print(f"Optimizer: {optimizer.__class__.__name__}")

# 定义学习率调度器
scheduler = nn.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=milestones,  # 迭代里程碑用于LR衰减
    gamma=0.5
)
print(f"Scheduler: {scheduler.__class__.__name__}")

# 定义损失函数（包含Softmax）
loss_fn = nn.op.CrossEntropyLoss(
    model=model,
    num_classes=int(train_labs.max() + 1)  # 对于MNIST应该是10
)
print(f"Loss function: {loss_fn.__class__.__name__}")

# --- Runner设置 ---
runner = nn.runner.RunnerM(
    model=model,
    optimizer=optimizer,
    metric=nn.metric.accuracy,
    loss_fn=loss_fn,
    scheduler=scheduler,
    batch_size=batch_size
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
fig.suptitle(f"{model_type.upper()} Training History")  # 根据模型类型设置标题
plot(runner, axes)  # Pass runner and axes to the plotting function
# Adjust layout to prevent title overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{model_type}_training_history.png")
print("Script finished.")
