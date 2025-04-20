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
