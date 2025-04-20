# Neural Network Implementation Report

## Implementation of Linear Layer

The linear layer is a fundamental building block in neural networks, performing an affine transformation of its inputs. In this report, I'll explain the mathematics behind the forward and backward propagation processes in a linear layer.

### Forward Propagation

The forward propagation in a linear layer applies a linear transformation to the input. Given an input $X$ with shape [batch_size, in_dim], the linear layer computes an output $Y$ with shape [batch_size, out_dim] as follows:

$$Y = XW + b$$

where:
- $X$ is the input matrix of shape [batch_size, in_dim]
- $W$ is the weight matrix of shape [in_dim, out_dim]
- $b$ is the bias vector of shape [1, out_dim]
- $Y$ is the output matrix of shape [batch_size, out_dim]

The implementation in code is straightforward:

```python
def forward(self, X):
    self.input = X  # Store for backward pass
    output = np.dot(X, self.W) + self.b
    return output
```

### Backward Propagation

During the backward propagation phase, we compute the gradients of the loss with respect to the inputs and parameters. The mathematical derivation follows the chain rule of calculus.

Given the gradient of the loss with respect to the output $\frac{\partial L}{\partial Y}$ (denoted as `grad` in the code), we compute:

1. **Gradient with respect to the input $X$:**
   $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$$

2. **Gradient with respect to the weights $W$:**
   $$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$$

3. **Gradient with respect to the bias $b$:**
   $$\frac{\partial L}{\partial b} = \sum_{i=1}^{B} \frac{\partial L}{\partial Y_i}$$

The implementation in code follows these equations:

```python
def backward(self, grad):
    # Gradient with respect to W
    dW = np.dot(self.input.T, grad)
    
    # Gradient with respect to b
    db = np.sum(grad, axis=0, keepdims=True)
    
    # Gradient with respect to X
    dX = np.dot(grad, self.W.T)
    
    # Store gradients for optimizer
    self.grads['W'] = dW
    self.grads['b'] = db
    
    return dX
```

### Weight Decay (L2 Regularization)

The linear layer also supports weight decay, which is equivalent to L2 regularization. When weight decay is enabled, we add an additional term to the gradient of the weights:

$$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y} + \lambda W$$

where $\lambda$ is the weight decay coefficient. This helps prevent overfitting by penalizing large weight values. In the code, this is implemented as:

```python
if self.weight_decay:
    self.grads['W'] += self.weight_decay_lambda * self.W
```

## Implementation of MultiCrossEntropyLoss

For my neural network implementation, I also needed to implement the Cross-Entropy Loss function with integrated Softmax functionality. This combination is extremely common in classification tasks where we need to convert network outputs into probabilities and then calculate the loss between predicted probabilities and true labels.

### Understanding the Math

The MultiCrossEntropyLoss combines two key components:

1. **Softmax Function**: Converts raw network outputs (logits) into probabilities
2. **Cross-Entropy Loss**: Measures the difference between predicted probability distribution and actual distribution

The softmax function transforms a vector of real numbers into a probability distribution:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

where $C$ is the number of classes. This ensures all outputs are between 0 and 1 and sum to 1.

The cross-entropy loss for multi-class classification is defined as:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} y_{ij} \log(p_{ij})$$

where:
- $N$ is the batch size
- $C$ is the number of classes
- $y_{ij}$ is 1 if sample $i$ belongs to class $j$ and 0 otherwise (one-hot encoding)
- $p_{ij}$ is the predicted probability that sample $i$ belongs to class $j$

### Numerical Stability

In my implementation, I needed to be careful about numerical stability. The softmax function involves exponentials that can explode, and the log function in the cross-entropy loss can't handle zero inputs. I addressed these issues by:

1. Subtracting the maximum value before computing exponentials in softmax (implemented in the provided softmax function)
2. Adding a small epsilon to prevent log(0) in the loss calculation:

```python
epsilon = 1e-10
log_probs = np.log(self.softmax_output + epsilon)
```

### Backward Pass (Gradient Calculation)

The really cool thing I discovered while implementing this is the elegant gradient formulation when softmax and cross-entropy are combined. For the softmax cross-entropy loss, the gradient with respect to the inputs is simply:

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

where $p_i$ is the softmax output and $y_i$ is the true label (in one-hot form). This elegant form arises from the cancellation of terms when we apply the chain rule through both the softmax and cross-entropy functions.

In my code, this gradient calculation looks like:

```python
# Gradient of softmax cross-entropy: softmax_output - one_hot_labels
self.grads = (self.softmax_output - one_hot_labels) / batch_size
```

I divide by the batch size to get the average gradient, which helps stabilize training.

### Flexible Implementation

I designed the implementation to be flexible, allowing the softmax layer to be optionally disabled through the `cancel_softmax` method:

```python
def cancel_soft_max(self):
    self.has_softmax = False
    return self
```

This would be useful if, for example, the model already applies a softmax activation or if we want to use different activation functions with the cross-entropy loss.

I learned a lot from implementing this loss function, especially about the elegant mathematics behind neural network training and why the softmax + cross-entropy combination is so widely used in classification problems.

## Implementation of Convolutional Layer

After implementing basic neural network components like the Linear layer and the MultiCrossEntropyLoss, I moved on to implement a 2D convolutional layer (conv2D). This was quite challenging but also fascinating because convolutional neural networks (CNNs) are responsible for so many breakthroughs in computer vision.

### Understanding Convolution Operations

The key idea behind convolution is to apply filters (also called kernels) to an input to extract features. Unlike a fully connected layer where every input neuron connects to every output neuron, convolutional layers create local connectivity patterns that are replicated across the entire input.

A single 2D convolution operation can be expressed mathematically as:

$$O[o, h, w] = \sum_{i=0}^{C_{in}-1} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} I[i, h \cdot s_h + k_h, w \cdot s_w + k_w] \cdot W[o, i, k_h, k_w] + b[o]$$

where:
- $O[h, w]$ is the output at position $(h, w)$ for a single channel
- $I$ is the input tensor
- $W$ is the weight tensor (filter/kernel)
- $b$ is the bias
- $C_{in}$ is the number of input channels
- $K_h, K_w$ are the kernel height and width
- $s_h, s_w$ are the stride values for height and width

### Implementing the Forward Pass

The forward pass in a convolutional layer involves scanning the input with filters and computing the dot product at each position. I implemented this using nested loops rather than any specialized functions for clarity:

```python
def forward(self, X):
    # Store input for backward pass
    self.input = X
    
    # Apply padding if specified
    # ...padding code...
    
    # Calculate output dimensions
    output_height = (padded_height - self.kernel_size[0]) // self.stride[0] + 1
    output_width = (padded_width - self.kernel_size[1]) // self.stride[1] + 1
    
    # Initialize output
    output = np.zeros((batch_size, self.out_channels, output_height, output_width))
    
    # Perform convolution operation
    for b in range(batch_size):
        for oc in range(self.out_channels):
            for h in range(output_height):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                
                for w in range(output_width):
                    w_start = w * self.stride[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    # Extract the current patch from the input
                    patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                    
                    # Perform convolution (element-wise multiplication and sum)
                    output[b, oc, h, w] = np.sum(patch * self.W[oc]) + self.b[oc, 0, 0]
```

I found it challenging to keep track of all the dimensions, especially when dealing with padding and strides. The key insight was to visualize the sliding window operation and carefully index into the tensors.

### Backpropagation in Convolutional Layers

The backward pass for a convolutional layer is more complex. Conceptually, we need to compute three gradients:

1. Gradient with respect to the input ($\frac{\partial L}{\partial X}$)
2. Gradient with respect to the weights ($\frac{\partial L}{\partial W}$)
3. Gradient with respect to the bias ($\frac{\partial L}{\partial b}$)

For the bias, it's relatively simple - we just sum the gradients across the batch and spatial dimensions.

For the weights, we need to perform a correlation operation between the input patches and the output gradients:

$$\frac{\partial L}{\partial W_{o,i,k_h,k_w}} = \sum_{b=0}^{B-1} \sum_{h=0}^{H_{out}-1} \sum_{w=0}^{W_{out}-1} I_{b,i,h \cdot s_h + k_h, w \cdot s_w + k_w} \cdot \frac{\partial L}{\partial O_{b,o,h,w}}$$

For the input, we need to perform a full convolution (with padding) of the output gradients with the flipped kernels:

$$\frac{\partial L}{\partial I_{b,i,h,w}} = \sum_{o=0}^{C_{out}-1} \sum_{k_h=0}^{K_h-1} \sum_{k_w=0}^{K_w-1} W_{o,i,k_h,k_w} \cdot \frac{\partial L}{\partial O_{b,o,\lfloor\frac{h-k_h}{s_h}\rfloor,\lfloor\frac{w-k_w}{s_w}\rfloor}}$$

My implementation of the backward pass uses nested loops to compute these gradients:

```python
def backward(self, grads):
    # Initialize gradients for weights, bias, and input
    dW = np.zeros_like(self.W)
    db = np.zeros_like(self.b)
    dX_padded = np.zeros_like(self.input_padded)
    
    # Calculate gradients
    for b in range(batch_size):
        for oc in range(self.out_channels):
            for h in range(output_height):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                
                for w in range(output_width):
                    w_start = w * self.stride[1]
                    w_end = w_start + self.kernel_size[1]
                    
                    # Extract the current patch from the input
                    patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                    
                    # Calculate gradient for weights
                    dW[oc] += patch * grads[b, oc, h, w]
                    
                    # Calculate gradient for bias
                    db[oc, 0, 0] += grads[b, oc, h, w]
                    
                    # Calculate gradient for input
                    dX_padded[b, :, h_start:h_end, w_start:w_end] += self.W[oc] * grads[b, oc, h, w]
```

### Reflection

Implementing the convolutional layer was definitely more challenging than the linear layer, primarily due to:

1. The complexity of managing multidimensional tensors
2. Handling padding correctly
3. Computing gradients with stride > 1
4. Making sure the dimensions aligned correctly in forward and backward passes

I found the most difficult aspect was getting the backward pass right. It required careful derivation of the gradients and a solid understanding of how convolution operations work mathematically.

While my implementation is not optimized for performance (it uses naive loops instead of vectorized operations), it helped me build a deeper understanding of CNNs. In real-world applications, we would use libraries like PyTorch or TensorFlow that implement highly optimized convolution operations, but implementing it from scratch was an invaluable learning experience.

I gained a much better appreciation for why CNNs work so well for image processing tasks - the parameter sharing and local connectivity are powerful inductive biases that drastically reduce the number of parameters compared to fully connected networks while preserving spatial information.
