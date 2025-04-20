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
