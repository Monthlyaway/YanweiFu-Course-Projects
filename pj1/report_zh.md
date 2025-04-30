# 神经网络实现报告

## 线性层实现

线性层是神经网络中的基本构建模块，执行输入的仿射变换。

### 前向传播

线性层中的前向传播对输入应用线性变换。给定形状为[batch_size, in_dim]的输入$X$，线性层计算形状为[batch_size, out_dim]的输出$Y$，如下所示：

$$Y = XW + b$$

其中：
- $X$是形状为[batch_size, in_dim]的输入矩阵
- $W$是形状为[in_dim, out_dim]的权重矩阵
- $b$是形状为[1, out_dim]的偏置向量
- $Y$是形状为[batch_size, out_dim]的输出矩阵

代码实现非常直接：

```python
def forward(self, X):
    self.input = X  # 存储用于反向传播
    output = np.dot(X, self.W) + self.b
    return output
```

### 反向传播

在反向传播阶段，我们计算损失相对于输入和参数的梯度。数学推导遵循微积分的链式法则。

给定损失相对于输出$\frac{\partial L}{\partial Y}$的梯度（在代码中表示为`grad`），我们计算：

1. **相对于输入$X$的梯度：**
   $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$$

2. **相对于权重$W$的梯度：**
   $$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$$

3. **相对于偏置$b$的梯度：**
   $$\frac{\partial L}{\partial b} = \sum_{i=1}^{B} \frac{\partial L}{\partial Y_i}$$

代码实现遵循这些方程：

```python
def backward(self, grad):
    # 相对于W的梯度
    dW = np.dot(self.input.T, grad)
    
    # 相对于b的梯度
    db = np.sum(grad, axis=0, keepdims=True)
    
    # 相对于X的梯度
    dX = np.dot(grad, self.W.T)
    
    # 存储梯度供优化器使用
    self.grads['W'] = dW
    self.grads['b'] = db
    
    return dX
```

### Weight Decay（L2正则化）

线性层还支持Weight Decay，这等同于L2正则化。启用Weight Decay后，我们向权重的梯度中添加一个额外项：

$$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y} + \lambda W$$

其中$\lambda$是权重衰减系数。这通过惩罚较大的权重值来帮助防止过拟合。在代码中，实现如下：

```python
if self.weight_decay:
    self.grads['W'] += self.weight_decay_lambda * self.W
```

## Multi-Class Cross-Entropy Loss

Multi-Class Cross-Entropy Loss结合了Softmax功能，这在分类任务中非常常见，需要将网络输出转换为概率并计算预测概率与真实标签之间的损失。

### 数学原理

Multi-Class Cross-Entropy Loss结合了两个关键组件：

1. **Softmax函数**：将原始网络输出（logits）转换为概率
2. **交叉熵损失**：测量预测概率分布与实际分布之间的差异

Softmax函数将实数向量转换为概率分布：

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

其中$C$是类别数量。这确保所有输出在0到1之间且总和为1。

多类分类的交叉熵损失定义为：

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} y_{ij} \log(p_{ij})$$

其中：
- $N$是批量大小
- $C$是类别数量
- $y_{ij}$在样本$i$属于类别$j$时为1，否则为0（one-hot encoding）
- $p_{ij}$是样本$i$属于类别$j$的预测概率

两个求和可以理解为对于每一个样本，我的预测分布和实际的one hot 分布的差别。

### 防止溢出

实现中需要注意溢出。Softmax函数涉及可能爆炸的指数运算，而交叉熵损失中的对数函数不能处理零输入。解决这些问题的方法是：

1. 在计算Softmax中的指数前减去最大值（在提供的Softmax函数中实现）
2. 在损失计算中添加一个小的epsilon以防止log(0)：

```python
epsilon = 1e-10
log_probs = np.log(self.softmax_output + epsilon)
```

### 反向传播

当Softmax和交叉熵组合时，梯度计算非常优雅。对于Softmax交叉熵损失，相对于输入的梯度简单地为：

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

其中$p_i$是Softmax输出，$y_i$是真实标签（one hot encoding）。这种优雅的形式来自于通过Softmax和交叉熵函数应用链式法则时项的消除。

在代码中，梯度计算如下：

```python
# Softmax交叉熵的梯度：softmax_output - one_hot_labels
self.grads = (self.softmax_output - one_hot_labels) / batch_size
```

梯度除以批量大小以获得平均梯度，或者也可以在之后每个layer的梯度更新中除以batch_size，但是这样就麻烦了。

## 卷积层

卷积神经网络（CNN）在计算机视觉中实现了许多突破。本节讲解二维卷积层（conv2D）的实现。

### 卷积操作

卷积背后的核心思想是对输入应用滤波器（也称为核）来提取特征。与全连接层中每个输入神经元连接到每个输出神经元不同，卷积层创建了在整个输入上复制的局部连接模式。

单个二维卷积操作可以数学表示为：

$$O[o, h, w] = \sum_{i=0}^{C_{in}-1} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} I[i, h \cdot s_h + k_h, w \cdot s_w + k_w] \cdot W[o, i, k_h, k_w] + b[o]$$

其中：
- $O[h, w]$是单个通道位置$(h, w)$的输出
- $I$是输入张量
- $W$是卷积核
- $b$是偏置
- $C_{in}$是输入通道数
- $K_h, K_w$是核的高度和宽度
- $s_h, s_w$是高度和宽度的步长值

### 前向传播

卷积层中的前向传播用卷积核扫描输入并在每个位置计算点积：

```python
def forward(self, X):
    # 存储输入用于反向传播
    self.input = X
    
    # 如果指定了则应用填充
    # ...填充代码...
    
    # 计算输出维度
    output_height = (padded_height - self.kernel_size[0]) // self.stride[0] + 1
    output_width = (padded_width - self.kernel_size[1]) // self.stride[1] + 1
    
    # 初始化输出
    output = np.zeros((batch_size, self.out_channels, output_height, output_width))
    
    # 执行卷积操作
    for b in range(batch_size):
        for oc in range(self.out_channels):
            for h in range(output_height):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                
                for w in range(output_width):
                    w_start = w * self.stride[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    # 从输入中提取当前块
                    patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                    
                    # 执行卷积（元素乘法和求和）
                    output[b, oc, h, w] = np.sum(patch * self.W[oc]) + self.b[oc, 0, 0]
```

### 卷积层中的反向传播

卷积层的反向传播更为复杂。概念上，我们需要计算三个梯度：

1. 相对于输入的梯度 ($\frac{\partial L}{\partial X}$)
2. 相对于权重的梯度 ($\frac{\partial L}{\partial W}$)
3. 相对于偏置的梯度 ($\frac{\partial L}{\partial b}$)

对于偏置，计算相对简单 - 我们只需在批次和空间维度上对梯度求和。

对于权重，我们需要在输入块和输出梯度之间执行相关操作：

$$\frac{\partial L}{\partial W_{o,i,k_h,k_w}} = \sum_{b=0}^{B-1} \sum_{h=0}^{H_{out}-1} \sum_{w=0}^{W_{out}-1} I_{b,i,h \cdot s_h + k_h, w \cdot s_w + k_w} \cdot \frac{\partial L}{\partial O_{b,o,h,w}}$$

对于输入，我们需要用翻转的核对输出梯度执行完全卷积（带填充）：

$$\frac{\partial L}{\partial I_{b,i,h,w}} = \sum_{o=0}^{C_{out}-1} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} W_{o,i,k_h,k_w} \cdot \frac{\partial L}{\partial O_{b,o,\lfloor\frac{h-k_h}{s_h}\rfloor,\lfloor\frac{w-k_w}{s_w}\rfloor}}$$

反向传播的实现使用嵌套循环来计算这些梯度：

```python
def backward(self, grads):
    # 初始化权重、偏置和输入的梯度
    dW = np.zeros_like(self.W)
    db = np.zeros_like(self.b)
    dX_padded = np.zeros_like(self.input_padded)
    
    # 计算梯度
    for b in range(batch_size):
        for oc in range(self.out_channels):
            for h in range(output_height):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                
                for w in range(output_width):
                    w_start = w * self.stride[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    # 从输入中提取当前块
                    patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                    
                    # 计算权重的梯度
                    dW[oc] += patch * grads[b, oc, h, w]
                    
                    # 计算偏置的梯度
                    db[oc, 0, 0] += grads[b, oc, h, w]
                    
                    # 计算输入的梯度
                    dX_padded[b, :, h_start:h_end, w_start:w_end] += self.W[oc] * grads[b, oc, h, w]
```

## Scheduler  


### 调度器基类

首先，为所有调度器创建一个基类：

```python
class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
        self.initial_lr = optimizer.init_lr

    @abstractmethod
    def step(self):
        pass

    def get_lr(self):
        return self.optimizer.init_lr
```

这种设计遵循面向对象原则，所有调度器都继承自具有一致接口的公共基类。

### StepLR 

实现的最简单的学习率调度器是StepLR，它每固定步数通过乘法因子减少学习率：

```python
def step(self) -> None:
    self.step_count += 1
    if self.step_count % self.step_size == 0:
        new_lr = self.optimizer.init_lr * self.gamma
        self.optimizer.init_lr = new_lr
```

数学上，这可以表达为：

$$\text{lr}_{\text{epoch}} = \text{lr}_{\text{initial}} \times \gamma^{\lfloor\frac{\text{epoch}}{\text{step\_size}}\rfloor}$$

其中：
- $\text{lr}_{\text{epoch}}$是给定epoch的学习率
- $\text{lr}_{\text{initial}}$是初始学习率
- $\gamma$是衰减因子（通常为0.1）
- $\text{step\_size}$是学习率衰减之间的epoch数

例如，初始学习率为0.1，step_size为30，gamma为0.1，学习率会是：
- Epoch 0-29：0.1
- Epoch 30-59：0.01
- Epoch 60-89：0.001
- 以此类推...

这在图表上会形成阶梯状模式。

### MultiStepLR

在某些情况下，常规的阶梯衰减可能不是最优的。有时需要根据训练动态在特定epoch减少学习率。这时MultiStepLR很有用：

```python
def step(self) -> None:
    self.step_count += 1
    if self.step_count in self.milestones and self.step_count > self.last_lr_update_step:
        new_lr = self.optimizer.init_lr * self.gamma
        self.optimizer.init_lr = new_lr
        self.last_lr_update_step = self.step_count
```

学习率遵循以下模式：

$$\text{lr}_{\text{epoch}} = \text{lr}_{\text{initial}} \times \gamma^{j}$$

其中$j$是已达到的里程碑数量。

例如，里程碑在epoch [30, 60, 80]：
- Epoch 0-29：初始学习率
- Epoch 30-59：初始学习率 × gamma
- Epoch 60-79：初始学习率 × gamma²
- Epoch 80+：初始学习率 × gamma³

这提供了更多的灵活性，可以根据训练过程定义自定义的衰减计划。

### 指数学习率衰减

最后，实现了ExponentialLR以获得更平滑的衰减模式：

```python
def step(self) -> None:
    self.step_count += 1
    new_lr = self.initial_lr * (self.gamma ** self.step_count)
    self.optimizer.init_lr = new_lr
```

学习率遵循这个指数衰减公式：

$$\text{lr}_{\text{epoch}} = \text{lr}_{\text{initial}} \times \gamma^{\text{epoch}}$$

这创建了一条平滑曲线，学习率持续减少而不是离散步骤。gamma值接近1（例如0.95）时，衰减更加渐进。

## Momentum 梯度下降

在实现了各种神经网络组件后，我们关注优化方法。虽然标准随机梯度下降（SGD）被广泛使用，但它在某些类型的损失景观中会遇到困难。动量梯度下降可以显著改善模型的训练行为。

### 标准SGD的限制

在标准SGD中，参数直接在负梯度方向上更新：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

其中$\theta$表示模型参数，$\eta$是学习率，$\nabla L(\theta_t)$是损失函数的梯度。

标准SGD存在几个问题：

1. **在valley中进展缓慢**：当损失表面在某些方向有陡峭斜坡而在其他方向有浅斜坡时（想象一个狭长的山谷），由于在陡峭维度上的振荡，SGD在浅维度上只能迈出小步。

2. **陷入局部极小值或鞍点**：没有动量，SGD可能轻易被困在次优解中。

3. **对学习率敏感**：找到正确的学习率至关重要但具有挑战性。太高，SGD会发散；太低，它会极其缓慢地收敛。

### 动量解决方案

动量梯度下降的更新方程是：

$$v_t = \mu v_{t-1} - \eta \nabla L(\theta_{t-1})$$
$$\theta_t = \theta_{t-1} + v_t$$

其中$\mu$是动量保留系数（通常约为0.9），$v_t$是速度向量。

### 动量的实现细节

动量梯度下降的实现需要跟踪所有参数的速度：

```python
def __init__(self, init_lr, model, mu=0.9):
    super().__init__(init_lr, model)
    self.mu = mu
    # 初始化速度字典
    self.velocities = {} 
    for i, layer in enumerate(self.model.layers):
         if hasattr(layer, 'optimizable') and layer.optimizable:
            self.velocities[i] = {}
            for key in layer.params.keys():
                # 将速度初始化为零
                self.velocities[i][key] = np.zeros_like(layer.params[key])
```

核心更新逻辑遵循动量更新规则：

```python
def step(self):
    for i, layer in enumerate(self.model.layers):
         if hasattr(layer, 'optimizable') and layer.optimizable:
            for key in layer.params.keys():
                # 计算梯度（如果需要则包括权重衰减）
                grad_update = layer.grads[key]
                if hasattr(layer, 'weight_decay') and layer.weight_decay:
                    grad_update += layer.weight_decay_lambda * layer.params[key]

                # 更新速度：v = mu * v - lr * grad
                self.velocities[i][key] = self.mu * self.velocities[i][key] - self.init_lr * grad_update

                # 更新参数：param = param + v
                layer.params[key] += self.velocities[i][key]
```

## Padding层

在卷积神经网络中，Padding（填充）是一种重要的操作，用于控制卷积输出的空间维度并保留边缘信息。

### Padding的原理

Padding的主要目的是在输入特征图的周围添加额外的像素（通常是零），以便：

1. **保持空间维度**：在应用卷积操作后保持输出特征图的空间维度与输入相同。
2. **保留边缘信息**：没有填充时，边缘像素在卷积过程中参与的次数比中心像素少，导致边缘信息丢失。
3. **提高特征提取质量**：通过保留边缘信息，可以提高模型对图像边缘特征的提取能力。

对于一个形状为 [batch_size, channels, height, width] 的输入张量，填充操作会在高度和宽度维度的两侧添加指定数量的零值像素，从而产生一个更大的输出张量。

### 前向传播

Padding层的前向传播非常直接，使用NumPy的pad函数实现：

```python
def forward(self, X):
    # 对高度和宽度进行padding，不对batch和channel维度padding
    output = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad),
                    (self.pad, self.pad)), 'constant', constant_values=0)
    return output
```

这里的填充模式是'constant'，填充值为0，这也被称为"零填充"。填充只应用于空间维度（高度和宽度），而不是批次大小或通道维度。

### 反向传播

在反向传播过程中，Padding层需要将梯度传回到原始未填充的输入。这意味着我们需要从填充后的梯度中提取出对应于原始输入的部分：

```python
def backward(self, grads):
    # 去除padding部分的梯度
    output = grads[:, :, self.pad:-self.pad, self.pad:-self.pad]
    return output
```

这个操作本质上是切片操作，移除了添加的填充部分的梯度，只保留原始输入区域的梯度。

## Pooling层

Pooling（池化）是卷积神经网络中另一个关键组件，用于减少特征图的空间维度，提取显著特征，并减少计算量。

### 池化的原理

池化操作通过在特征图上滑动一个固定大小的窗口，并从每个窗口区域提取一个代表性值来减少空间维度。最常用的池化方法是最大池化（Max Pooling），它从每个窗口中选择最大值。

池化操作的主要优势包括：

1. **降维**：减少特征图的空间维度，降低后续层的计算复杂度。
2. **特征提取**：通过选择最显著的特征（如最大值），增强模型对关键特征的关注。
3. **位置不变性**：对输入的小位移具有一定的鲁棒性，有助于模型泛化。

### 前向传播

最大池化的前向传播涉及在输入特征图上滑动窗口，并计算每个窗口内的最大值：

```python
def forward(self, X):
    self.input = X
    batch_size, channels, height, width = X.shape
    
    # 计算输出尺寸
    out_height = (height - self.pool_size) // self.stride + 1
    out_width = (width - self.pool_size) // self.stride + 1
    
    # 初始化输出和最大值索引
    output = np.zeros((batch_size, channels, out_height, out_width))
    self.max_indices = np.zeros(
        (batch_size, channels, out_height, out_width, 2), dtype=int)
    
    # 执行最大池化
    for b in range(batch_size):
        for c in range(channels):
            for h in range(out_height):
                h_start = h * self.stride
                h_end = h_start + self.pool_size
                
                for w in range(out_width):
                    w_start = w * self.stride
                    w_end = w_start + self.pool_size
                    
                    # 提取当前区域
                    pool_region = X[b, c, h_start:h_end, w_start:w_end]
                    
                    # 找到最大值
                    output[b, c, h, w] = np.max(pool_region)
                    
                    # 记录最大值的位置（相对于池化窗口）
                    max_idx = np.unravel_index(
                        np.argmax(pool_region), pool_region.shape)
                    self.max_indices[b, c, h, w] = max_idx
    
    return output
```

在这个实现中，我们不仅计算了最大值，还记录了每个最大值在原始池化窗口中的位置（存储在`max_indices`中）。这些索引在反向传播中至关重要。

### 反向传播

池化层的反向传播是一个稀疏操作。由于在前向传播中，每个输出值只来自一个输入值（最大值），因此梯度只需要传回到这些特定位置：

```python
def backward(self, grads):
    batch_size, channels, out_height, out_width = grads.shape
    
    # 初始化输入梯度
    dX = np.zeros_like(self.input)
    
    # 反向传播梯度
    for b in range(batch_size):
        for c in range(channels):
            for h in range(out_height):
                h_start = h * self.stride
                
                for w in range(out_width):
                    w_start = w * self.stride
                    
                    # 获取最大值的位置
                    max_h, max_w = self.max_indices[b, c, h, w]
                    
                    # 将梯度传递给最大值位置
                    dX[b, c, h_start + max_h, w_start + max_w] += grads[b, c, h, w]
    
    return dX
```

在这个过程中，我们使用存储的最大值索引来确定应该将梯度传递到输入的哪个位置。这种方法确保了梯度只流向那些在前向传播中被选为最大值的元素，而其他元素的梯度保持为零。

