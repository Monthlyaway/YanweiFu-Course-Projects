import matplotlib.pyplot as plt
import numpy as np

def plot_lr_curves(iterations, lr_curves, labels, save_path):
    """
    绘制学习率变化曲线对比图
    
    Args:
        iterations: 迭代次数数组
        lr_curves: 不同调度器的学习率变化列表
        labels: 调度器名称列表
        save_path: 图像保存路径
    """
    plt.figure(figsize=(10, 6), dpi=600)
    
    # 设置不同调度器的颜色
    colors = ['#E3E37D', '#968A62', '#4C8C6F']
    
    for i, (lrs, label) in enumerate(zip(lr_curves, labels)):
        plt.plot(iterations, lrs, label=label, color=colors[i], linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # 使用对数坐标以更好地显示学习率变化
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_metrics(runners, labels, save_dir):
    """
    绘制不同调度器下的训练效果对比图
    
    Args:
        runners: 不同调度器对应的训练runner列表
        labels: 调度器名称列表
        save_dir: 图像保存目录
    """
    # 创建loss和accuracy的对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=600)
    
    colors = ['#E3E37D', '#968A62', '#4C8C6F']
    
    for i, (runner, label) in enumerate(zip(runners, labels)):
        epochs = range(len(runner.train_scores))
        
        # 绘制loss曲线
        ax1.plot(epochs, runner.train_loss, color=colors[i], 
                label=f"{label} (Train)", linewidth=2)
        ax1.plot(epochs, runner.dev_loss, color=colors[i], 
                linestyle="--", label=f"{label} (Dev)", linewidth=2)
        
        # 绘制accuracy曲线
        ax2.plot(epochs, runner.train_scores, color=colors[i], 
                label=f"{label} (Train)", linewidth=2)
        ax2.plot(epochs, runner.dev_scores, color=colors[i], 
                linestyle="--", label=f"{label} (Dev)", linewidth=2)
    
    # 设置图表属性
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_metrics.png")
    plt.close()

def save_metrics_report(runners, labels, save_path):
    """
    保存不同调度器的训练指标报告
    
    Args:
        runners: 不同调度器对应的训练runner列表
        labels: 调度器名称列表
        save_path: 报告保存路径
    """
    with open(save_path, 'w') as f:
        f.write("学习率调度器实验结果报告\n")
        f.write("======================\n\n")
        
        for runner, label in zip(runners, labels):
            f.write(f"\n{label}:\n")
            f.write("-" * len(label) + "\n")
            f.write(f"最终训练loss: {runner.train_loss[-1]:.4f}\n")
            f.write(f"最终验证loss: {runner.dev_loss[-1]:.4f}\n")
            f.write(f"最终训练accuracy: {runner.train_scores[-1]:.4f}\n")
            f.write(f"最终验证accuracy: {runner.dev_scores[-1]:.4f}\n")
            f.write(f"最佳验证accuracy: {max(runner.dev_scores):.4f}\n")
            f.write("\n")