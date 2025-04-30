from .op import *
import pickle
import logging

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """

    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_path):
        """
        Load model parameters from a file
        param_path: Path to the pickle file containing model parameters
        """
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        # Initialize the layers list
        self.layers = []

        # Create layers based on loaded parameters
        for i in range(len(self.size_list) - 1):
            # Create linear layer
            layer = Linear(
                in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']

            # Add linear layer
            self.layers.append(layer)

            # Add activation layer if not the last linear layer
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                    self.layers.append(layer_f)

    def save_model(self, save_path):
        """
        Save model parameters to a file
        save_path: Path where to save the pickle file
        """
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """

    def __init__(self, in_channels=1, filter_sizes=None, act_func=None, lambda_list=None):
        self.in_channels = in_channels
        self.filter_sizes = filter_sizes
        self.act_func = act_func

        if filter_sizes is not None and act_func is not None:
            self.layers = []
            current_channels = in_channels

            # 添加第一个卷积层和池化层
            if len(filter_sizes) > 0:
                # 添加padding层
                self.layers.append(Padding(pad=1))
                # 添加第一个卷积层
                conv1 = conv2D(in_channels=current_channels, out_channels=filter_sizes[0],
                               kernel_size=3, stride=1, padding=0)
                if lambda_list is not None:
                    conv1.weight_decay = True
                    conv1.weight_decay_lambda = lambda_list[0]
                self.layers.append(conv1)

                # 添加池化层
                self.layers.append(Pooling(pool_size=2, stride=2))

                # 添加激活函数
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")

                current_channels = filter_sizes[0]

            # 添加第二个卷积层和池化层
            if len(filter_sizes) > 1:
                # 添加padding层
                self.layers.append(Padding(pad=1))
                # 添加第二个卷积层
                conv2 = conv2D(in_channels=current_channels, out_channels=filter_sizes[1],
                               kernel_size=3, stride=1, padding=0)
                if lambda_list is not None:
                    conv2.weight_decay = True
                    conv2.weight_decay_lambda = lambda_list[1]
                self.layers.append(conv2)

                # 添加池化层
                self.layers.append(Pooling(pool_size=2, stride=2))

                # 添加激活函数
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")

                current_channels = filter_sizes[1]

            # 添加Reshape层
            self.layers.append(Reshape())

            # 添加全连接层
            if len(filter_sizes) > 2:
                # 计算经过两次池化后的特征图大小 (28/2/2 = 7)
                feature_size = 7 * 7 * current_channels

                # 添加第一个全连接层
                fc1 = Linear(in_dim=feature_size, out_dim=filter_sizes[2])
                if lambda_list is not None and len(lambda_list) > 2:
                    fc1.weight_decay = True
                    fc1.weight_decay_lambda = lambda_list[2]
                self.layers.append(fc1)

                # 添加激活函数
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")

                # 添加输出层
                if len(filter_sizes) > 3:
                    fc2 = Linear(
                        in_dim=filter_sizes[2], out_dim=filter_sizes[3])
                    if lambda_list is not None and len(lambda_list) > 3:
                        fc2.weight_decay = True
                        fc2.weight_decay_lambda = lambda_list[3]
                    self.layers.append(fc2)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.filter_sizes is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with filter_sizes and act_func offered.'

        # 确保输入是4D张量 [batch_size, channels, height, width]
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        elif len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, int(
                np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))

        logging.debug(f"[CNN Forward] Input shape: {X.shape}")
        outputs = X

        for i, layer in enumerate(self.layers):
            outputs = layer(outputs)
            logging.debug(
                f"[CNN Forward] After layer {i} ({layer.__class__.__name__}) shape: {outputs.shape}")

        return outputs

    def backward(self, loss_grad):
        logging.debug(f"[CNN Backward] Input gradient shape: {loss_grad.shape}")
        grads = loss_grad

        for i, layer in enumerate(reversed(self.layers)):
            grads = layer.backward(grads)
            logging.debug(
                f"[CNN Backward] After layer {len(self.layers)-i-1} ({layer.__class__.__name__}) gradient shape: {grads.shape}")

        return grads

    def load_model(self, param_path):
        """Load model parameters from a file"""
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)

        self.in_channels = param_list[0]
        self.filter_sizes = param_list[1]
        self.act_func = param_list[2]

        # Initialize the layers list
        self.layers = []

        # 重建模型结构
        self.__init__(self.in_channels, self.filter_sizes, self.act_func)

        # 加载参数
        param_idx = 3
        for layer in self.layers:
            if layer.optimizable:
                if hasattr(layer, 'W') and hasattr(layer, 'b'):
                    layer.W = param_list[param_idx]['W']
                    layer.b = param_list[param_idx]['b']
                    layer.params['W'] = layer.W
                    layer.params['b'] = layer.b
                    layer.weight_decay = param_list[param_idx]['weight_decay']
                    layer.weight_decay_lambda = param_list[param_idx]['lambda']
                    param_idx += 1

    def save_model(self, save_path):
        """Save model parameters to a file"""
        param_list = [self.in_channels, self.filter_sizes, self.act_func]

        for layer in self.layers:
            if layer.optimizable:
                if hasattr(layer, 'W') and hasattr(layer, 'b'):
                    param_list.append({
                        'W': layer.params['W'],
                        'b': layer.params['b'],
                        'weight_decay': layer.weight_decay,
                        'lambda': layer.weight_decay_lambda
                    })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Padding(Layer):
    """Padding layer for CNN"""

    def __init__(self, pad=1):
        super().__init__()
        self.pad = pad
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        logging.debug(f"[Padding Forward] Input shape: {X.shape}, pad: {self.pad}")
        # 对高度和宽度进行padding，不对batch和channel维度padding
        output = np.pad(X, ((0, 0), (0, 0), (self.pad, self.pad),
                        (self.pad, self.pad)), 'constant', constant_values=0)
        logging.debug(f"[Padding Forward] Output shape: {output.shape}")
        return output

    def backward(self, grads):
        logging.debug(
            f"[Padding Backward] Input gradient shape: {grads.shape}, pad: {self.pad}")
        # 去除padding部分的梯度
        output = grads[:, :, self.pad:-self.pad, self.pad:-self.pad]
        logging.debug(f"[Padding Backward] Output gradient shape: {output.shape}")
        return output


class Pooling(Layer):
    """Pooling layer for CNN"""

    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.max_indices = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        logging.debug(
            f"[Pooling Forward] Input shape: {X.shape}, pool_size: {self.pool_size}, stride: {self.stride}")
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

        logging.debug(f"[Pooling Forward] Output shape: {output.shape}")
        return output

    def backward(self, grads):
        logging.debug(f"[Pooling Backward] Input gradient shape: {grads.shape}")
        batch_size, channels, out_height, out_width = grads.shape
        _, _, in_height, in_width = self.input.shape

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
                        dX[b, c, h_start + max_h, w_start +
                            max_w] += grads[b, c, h, w]

        logging.debug(f"[Pooling Backward] Output gradient shape: {dX.shape}")
        return dX


class Reshape(Layer):
    """Reshape layer for CNN to flatten the output for fully connected layers"""

    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        logging.debug(f"[Reshape Forward] Input shape: {X.shape}")
        self.input_shape = X.shape
        output = X.reshape(X.shape[0], -1)
        logging.debug(f"[Reshape Forward] Output shape: {output.shape}")
        return output

    def backward(self, grads):
        logging.debug(f"[Reshape Backward] Input gradient shape: {grads.shape}")
        output = grads.reshape(self.input_shape)
        logging.debug(f"[Reshape Backward] Output gradient shape: {output.shape}")
        return output


# 测试CNN模型
if __name__ == "__main__":
    import numpy as np
    
    # 设置logging级别为DEBUG
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # 生成随机数据代替mnist数据集
    # 创建随机的图像数据，形状为[batch_size, height, width]
    batch_size = 32
    img_height = 28
    img_width = 28
    num_classes = 10
    
    # 随机生成图像数据
    x_sample = np.random.rand(batch_size, img_height, img_width)
    # 随机生成标签
    y_sample = np.random.randint(0, num_classes, size=batch_size)
    
    print("\n===== 使用随机生成的numpy数组测试CNN模型 =====")
    print(f"生成的随机数据形状: {x_sample.shape}")
    print(f"生成的随机标签形状: {y_sample.shape}")
    
    # 创建CNN模型
    cnn = Model_CNN(in_channels=1, filter_sizes=[2, 3, 64, 10], act_func='ReLU')
    
    # 前向传播测试
    print("\n===== 前向传播测试 =====")
    output = cnn.forward(x_sample)
    print(f"\n最终输出形状: {output.shape}")
    
    # 创建损失函数
    loss_fn = CrossEntropyLoss(model=cnn, num_classes=num_classes)
    loss = loss_fn(output, y_sample)
    print(f"计算的损失值: {loss}")
    
    # 反向传播测试
    print("\n===== 反向传播测试 =====")
    loss_fn.backward()
    
    print("\nCNN模型测试完成！")
