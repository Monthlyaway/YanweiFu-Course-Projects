import os
import numpy as np
import matplotlib.pyplot as plt
import gzip
from struct import unpack
import pickle
import json
from tqdm import tqdm

import mynn as nn
from mynn.models import Model_MLP
from mynn.optimizer import SGD, MomentGD, AlternativeMomentGD
from mynn.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
from mynn.metric import accuracy

# 固定随机种子以确保实验可重复
np.random.seed(309)

# 创建保存结果的目录
RESULTS_DIR = 'experiment_results/mlp'
MODELS_DIR = 'saved_models/mlp'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 加载MNIST数据集


def load_mnist_data():
    data_dir = 'dataset/MNIST'
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')

    print(f"加载训练图像: {train_images_path}")
    with gzip.open(train_images_path, 'rb') as f:
        magic, num_images, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            num_images, rows * cols)

    print(f"加载训练标签: {train_labels_path}")
    with gzip.open(train_labels_path, 'rb') as f:
        magic, num_labels = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    # 打乱数据集
    idx = np.random.permutation(np.arange(num_images))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]

    # 分割为训练集和验证集
    num_validation = 10000
    valid_imgs = train_imgs[:num_validation]
    valid_labs = train_labs[:num_validation]
    train_imgs = train_imgs[num_validation:]
    train_labs = train_labs[num_validation:]

    # 归一化像素值从[0, 255]到[0, 1]
    train_imgs = train_imgs / 255.0
    valid_imgs = valid_imgs / 255.0

    print(f"训练样本数: {train_imgs.shape[0]}, 验证样本数: {valid_imgs.shape[0]}")
    return (train_imgs, train_labs), (valid_imgs, valid_labs)

# 创建MLP模型


def create_mlp_model():
    return Model_MLP(
        size_list=[784, 600, 10],  # [输入维度, 隐藏层维度, 输出维度]
        act_func='ReLU',
        lambda_list=[1e-4, 1e-4]  # 两层的权重衰减
    )

# 运行单个实验


def run_experiment(optimizer_name, scheduler_name, train_set, dev_set):
    print(f"\n开始实验: {optimizer_name} + {scheduler_name}")

    # 创建模型
    model = create_mlp_model()

    # 创建损失函数
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model)

    # 基础学习率
    init_lr = 0.01

    # 创建优化器
    if optimizer_name == 'SGD':
        optimizer = SGD(init_lr=init_lr, model=model)
    elif optimizer_name == 'MomentGD':
        optimizer = MomentGD(init_lr=init_lr, model=model, mu=0.9)
    # elif optimizer_name == 'AlternativeMomentGD':
    #     optimizer = AlternativeMomentGD(init_lr=init_lr, model=model, mu=0.9)
    else:
        raise ValueError(f"未知的优化器: {optimizer_name}")

    # 创建学习率调度器
    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.9)
    elif scheduler_name == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer=optimizer, milestones=[
                                1600, 4000, 10000], gamma=0.5)
    elif scheduler_name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer=optimizer, gamma=0.995)
    elif scheduler_name == 'None':
        scheduler = None
    else:
        raise ValueError(f"未知的学习率调度器: {scheduler_name}")

    # 创建训练器
    runner = nn.runner.RunnerM(
        model=model,
        optimizer=optimizer,
        metric=accuracy,
        loss_fn=loss_fn,
        batch_size=64,
        scheduler=scheduler
    )

    # 实验名称
    exp_name = f"{optimizer_name}_{scheduler_name}"
    save_dir = os.path.join(MODELS_DIR, exp_name)

    # 训练模型
    runner.train(
        train_set=train_set,
        dev_set=dev_set,
        num_epochs=30,  # 可以根据需要调整
        log_iters=30,
        save_dir=save_dir,
        scheduler_per_epoch=True,  # 每个epoch调整一次学习率
        early_stopping=10  # 10个epoch没有提升就提前停止
    )

    # 保存训练历史
    history = {
        'train_loss': runner.train_loss,
        'train_accuracy': runner.train_scores,
        'val_loss': runner.dev_loss,
        'val_accuracy': runner.dev_scores,
        'best_val_accuracy': runner.best_score,
        'best_epoch': runner.best_epoch
    }

    # 保存训练历史到JSON文件
    history_path = os.path.join(RESULTS_DIR, f"{exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)

    # 保存学习率历史（如果有调度器）
    if scheduler is not None:
        lr_history = []
        current_lr = init_lr
        for i in range(len(runner.train_loss)):
            if scheduler_name == 'StepLR':
                if (i+1) % 50 == 0:
                    current_lr *= 0.1
            elif scheduler_name == 'MultiStepLR':
                if (i+1) in [1600, 4000, 10000]:
                    current_lr *= 0.5
            elif scheduler_name == 'ExponentialLR':
                current_lr *= 0.995
            lr_history.append(current_lr)

        # 保存学习率历史到JSON文件
        lr_path = os.path.join(RESULTS_DIR, f"{exp_name}_lr.json")
        with open(lr_path, 'w') as f:
            json.dump(lr_history, f)

    print(f"实验 {exp_name} 完成")
    print(f"最佳验证准确率: {runner.best_score:.4f} (Epoch {runner.best_epoch})")
    return history

# 主函数


def main():
    print("开始MLP实验")

    # 加载数据
    train_set, dev_set = load_mnist_data()

    # 定义要测试的优化器和学习率调度器
    optimizers = ['SGD', 'MomentGD']
    schedulers = ['StepLR', 'MultiStepLR', 'ExponentialLR']

    # 存储所有实验结果
    all_results = {}

    # 运行所有实验组合
    for opt in optimizers:
        for sch in schedulers:
            exp_name = f"{opt}_{sch}"
            history = run_experiment(opt, sch, train_set, dev_set)
            all_results[exp_name] = {
                'best_val_accuracy': history['best_val_accuracy'],
                'best_epoch': history['best_epoch']
            }

    # 保存所有实验结果摘要
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f)

    print("\n所有MLP实验完成!")
    print("实验结果摘要:")
    for exp_name, result in all_results.items():
        print(
            f"{exp_name}: 最佳验证准确率 = {result['best_val_accuracy']:.4f} (Epoch {result['best_epoch']})")


if __name__ == "__main__":
    main()
