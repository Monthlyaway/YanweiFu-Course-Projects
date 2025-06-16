#!/usr/bin/env python3
"""
深度卷积生成对抗网络（DCGAN）完整训练脚本
基于Pokemon数据集的图像生成

作者：Chunyu Yang
描述：这个脚本展示了DCGAN的完整训练过程，包括数据下载、模型定义、训练循环和结果可视化
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import urllib.request
import zipfile
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 创建文件夹
os.makedirs('./results', exist_ok=True)

# ================== 数据集下载和准备 ==================


def download_pokemon_dataset(data_dir='./data'):
    """
    下载Pokemon精灵图数据集

    参数：
        data_dir: 数据存储目录
    """
    pokemon_url = 'http://d2l-data.s3-accelerate.amazonaws.com/pokemon.zip'
    pokemon_zip = os.path.join(data_dir, 'pokemon.zip')
    pokemon_dir = os.path.join(data_dir, 'pokemon')

    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)

    # 检查是否已经下载
    if not os.path.exists(pokemon_dir):
        print("正在下载Pokemon数据集...")
        # 下载数据集
        urllib.request.urlretrieve(pokemon_url, pokemon_zip)
        print("下载完成！正在解压...")

        # 解压数据集
        with zipfile.ZipFile(pokemon_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("解压完成！")

        # 删除zip文件
        os.remove(pokemon_zip)
    else:
        print("Pokemon数据集已存在，跳过下载。")

    return pokemon_dir

# ================== 生成器模块定义 ==================


class GeneratorBlock(nn.Module):
    """
    生成器基础模块
    使用转置卷积进行上采样，批归一化稳定训练，ReLU激活
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(GeneratorBlock, self).__init__()

        # 转置卷积层：将特征图尺寸扩大
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            bias=False  # 使用批归一化时不需要偏置
        )

        # 批归一化：稳定训练过程，加速收敛
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # ReLU激活：引入非线性
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    """
    DCGAN生成器
    将随机噪声向量转换为64x64的RGB图像
    """

    def __init__(self, noise_dim=100, feature_dim=64):
        super(Generator, self).__init__()

        # 渐进式生成：从1x1逐步扩大到64x64
        self.main = nn.Sequential(
            # 输入: (batch_size, 100, 1, 1)
            # 输出: (batch_size, 512, 4, 4)
            GeneratorBlock(noise_dim, feature_dim * 8, stride=1, padding=0),

            # 输出: (batch_size, 256, 8, 8)
            GeneratorBlock(feature_dim * 8, feature_dim * 4),

            # 输出: (batch_size, 128, 16, 16)
            GeneratorBlock(feature_dim * 4, feature_dim * 2),

            # 输出: (batch_size, 64, 32, 32)
            GeneratorBlock(feature_dim * 2, feature_dim),

            # 最后一层：转换为RGB图像
            # 输出: (batch_size, 3, 64, 64)
            nn.ConvTranspose2d(feature_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # 输出范围[-1, 1]
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重：使用均值0，标准差0.02的正态分布"""
        if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, noise):
        return self.main(noise)

# ================== 判别器模块定义 ==================


class DiscriminatorBlock(nn.Module):
    """
    判别器基础模块
    使用卷积进行下采样，批归一化稳定训练，LeakyReLU避免死神经元
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DiscriminatorBlock, self).__init__()

        # 卷积层：提取特征并减小尺寸
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            bias=False
        )

        # 批归一化
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # LeakyReLU：允许负值梯度流动，防止神经元"死亡"
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x


class Discriminator(nn.Module):
    """
    DCGAN判别器
    判断64x64的RGB图像是真实的还是生成的
    """

    def __init__(self, feature_dim=64):
        super(Discriminator, self).__init__()

        # 层次化特征提取：从64x64逐步压缩到1x1
        self.main = nn.Sequential(
            # 输入: (batch_size, 3, 64, 64)
            # 第一层不使用批归一化
            nn.Conv2d(3, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出: (batch_size, 128, 16, 16)
            DiscriminatorBlock(feature_dim, feature_dim * 2),

            # 输出: (batch_size, 256, 8, 8)
            DiscriminatorBlock(feature_dim * 2, feature_dim * 4),

            # 输出: (batch_size, 512, 4, 4)
            DiscriminatorBlock(feature_dim * 4, feature_dim * 8),

            # 最后一层：输出单个判断值
            # 输出: (batch_size, 1, 1, 1)
            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False)
            # 注意：不使用Sigmoid，因为使用BCEWithLogitsLoss
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, image):
        return self.main(image).view(-1, 1).squeeze(1)

# ================== 训练相关函数 ==================


def train_discriminator(discriminator, generator, real_images, noise,
                        criterion, optimizer, device):
    """
    训练判别器一步
    目标：正确分类真实图像为1，生成图像为0
    """
    batch_size = real_images.size(0)

    # 真实图像标签全为1
    real_labels = torch.ones(batch_size, device=device)
    # 生成图像标签全为0
    fake_labels = torch.zeros(batch_size, device=device)

    # 清空梯度
    optimizer.zero_grad()

    # 1. 训练判别真实图像
    real_output = discriminator(real_images)
    real_loss = criterion(real_output, real_labels)

    # 2. 训练判别生成图像
    fake_images = generator(noise)
    # detach()阻止梯度流向生成器
    fake_output = discriminator(fake_images.detach())
    fake_loss = criterion(fake_output, fake_labels)

    # 总损失 = 真实损失 + 生成损失
    d_loss = real_loss + fake_loss

    # 反向传播和优化
    d_loss.backward()
    optimizer.step()

    return d_loss.item(), real_output.mean().item(), fake_output.mean().item()


def train_generator(discriminator, generator, noise, criterion, optimizer, device):
    """
    训练生成器一步
    目标：让判别器将生成图像判断为真实（标签为1）
    """
    batch_size = noise.size(0)

    # 生成器希望判别器输出1
    real_labels = torch.ones(batch_size, device=device)

    # 清空梯度
    optimizer.zero_grad()

    # 生成假图像
    fake_images = generator(noise)

    # 判别器判断假图像
    fake_output = discriminator(fake_images)

    # 计算损失：希望判别器认为是真的
    g_loss = criterion(fake_output, real_labels)

    # 反向传播和优化
    g_loss.backward()
    optimizer.step()

    return g_loss.item()

# ================== 可视化函数 ==================


def save_samples(generator, fixed_noise, epoch, save_dir='./results'):
    """
    生成并保存样本图像
    """
    os.makedirs(save_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        # 反归一化：从[-1,1]转换到[0,1]
        fake_images = (fake_images + 1) / 2

        # 使用torchvision的工具函数创建图像网格
        image_grid = vutils.make_grid(fake_images, nrow=8, normalize=False)

        # 保存图像
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
        vutils.save_image(image_grid, save_path)

    generator.train()
    return image_grid


def plot_losses(d_losses, g_losses, save_path='./results/losses.png'):
    """
    绘制训练过程中的损失曲线
    """
    plt.figure(figsize=(10, 6))

    # 在同一个图中绘制判别器和生成器的损失曲线
    plt.plot(d_losses, label='Discriminator Loss', color='blue', alpha=0.7)
    plt.plot(g_losses, label='Generator Loss', color='red', alpha=0.7)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Discriminator and Generator Loss over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()


def show_results(generator, fixed_noise, num_epochs, d_losses, g_losses):
    """
    展示最终结果：仅展示生成的图像网格
    """
    # 生成最终图像
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        fake_images = (fake_images + 1) / 2  # 反归一化

    # 创建单个子图
    fig = plt.figure(figsize=(12, 10))

    # 显示生成的图像
    image_grid = vutils.make_grid(fake_images[:32], nrow=8, normalize=False)
    plt.imshow(np.transpose(image_grid.cpu().numpy(), (1, 2, 0)))
    plt.title(f'Generated Pokemon after {num_epochs} epochs')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('./results/final_results.png', bbox_inches='tight', dpi=600)
    plt.show()


# ================== 主训练函数 ==================
def train_dcgan(num_epochs=50, batch_size=128, learning_rate=0.0002,
                noise_dim=100, device='cuda', plot_interval=10):
    """
    DCGAN主训练函数

    参数：
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        noise_dim: 噪声向量维度
        device: 训练设备
        plot_interval: 绘制损失图像和生成图像网格的间隔
    """
    # 1. 准备设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 下载数据集
    data_dir = download_pokemon_dataset()

    # 3. 数据预处理
    # 关键：归一化到[-1, 1]以匹配生成器的tanh输出
    transform = transforms.Compose([
        transforms.Resize((64, 64)),           # 调整尺寸
        transforms.ToTensor(),                 # 转为张量并归一化到[0,1]
        transforms.Normalize((0.5, 0.5, 0.5),  # 标准化到[-1,1]
                             (0.5, 0.5, 0.5))
    ])

    # 4. 加载数据集
    dataset = torchvision.datasets.ImageFolder(
        root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=8, drop_last=True, persistent_workers=True, pin_memory=True)
    print(f"数据集大小: {len(dataset)} 张图像")

    # 5. 创建模型
    generator = Generator(noise_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)

    # 6. 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()

    # 关键参数：beta1=0.5而不是默认的0.9，让优化器更敏捷
    g_optimizer = torch.optim.Adam(generator.parameters(),
                                   lr=learning_rate / 10, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(),
                                   lr=learning_rate, betas=(0.5, 0.999))

    # 7. 固定噪声用于可视化进度
    fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)

    # 8. 记录训练历史
    d_losses = []
    g_losses = []
    d_real_scores = []
    d_fake_scores = []

    # 9. 开始训练
    print("开始训练...")
    total_iterations = 0

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for i, (real_images, _) in enumerate(progress_bar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 生成随机噪声
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)

            # ========== 训练判别器 ==========
            d_loss, d_real, d_fake = train_discriminator(
                discriminator, generator, real_images, noise,
                criterion, d_optimizer, device
            )

            # ========== 训练生成器 ==========
            # 生成新的噪声（重要：不重用之前的噪声）
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            g_loss = train_generator(
                discriminator, generator, noise,
                criterion, g_optimizer, device
            )

            # 记录损失
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            d_real_scores.append(d_real)
            d_fake_scores.append(d_fake)

            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
            total_iterations += 1

            # 更新进度条
            progress_bar.set_postfix({
                'D_loss': f'{d_loss:.4f}',
                'G_loss': f'{g_loss:.4f}',
                'D(x)': f'{d_real:.4f}',
                'D(G(z))': f'{d_fake:.4f}'
            })

        # 每过指定间隔的 epoch 或最后一个 epoch 时，绘制损失图像和生成图像网格
        if epoch % plot_interval == 0 or epoch == num_epochs - 1:
            save_samples(generator, fixed_noise, epoch + 1)
            plot_losses(d_losses, g_losses,
                        save_path=f'./results/losses_epoch_{epoch + 1:03d}.png')
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'D_loss: {epoch_d_loss/len(dataloader):.4f}, '
                  f'G_loss: {epoch_g_loss/len(dataloader):.4f}')

    # 10. 保存最终模型
    torch.save(generator.state_dict(), './results/generator_final.pth')
    torch.save(discriminator.state_dict(), './results/discriminator_final.pth')
    print("模型已保存！")

    # 11. 绘制最终损失曲线
    plot_losses(d_losses, g_losses, save_path='./results/losses_final.png')

    # 12. 展示最终结果
    show_results(generator, fixed_noise, num_epochs, d_losses, g_losses)

    return generator, discriminator


# ================== 主程序入口 ==================
if __name__ == "__main__":
    # 设置超参数
    config = {
        'num_epochs': 101,        # 训练轮数（可以增加以获得更好效果）
        'batch_size': 128,       # 批次大小
        'learning_rate': 0.0002,  # 学习率（DCGAN推荐值）
        'noise_dim': 100,        # 噪声维度
        'device': 'cuda',         # 使用GPU（如果可用）
        'plot_interval': 10       # 绘制损失图像和生成图像网格的间隔
    }

    # 开始训练
    generator, discriminator = train_dcgan(**config)

    print("\n训练完成！")
    print("生成的图像保存在 ./results/ 目录下")
    print("损失曲线保存为 ./results/losses_*.png")
    print("最终结果保存为 ./results/final_results.png")
