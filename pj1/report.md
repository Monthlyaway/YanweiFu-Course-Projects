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

## Learning Rate Schedulers

Adjusting the learning rate during training is a common technique to improve convergence and final model performance. I implemented a few learning rate schedulers to experiment with this. The idea is to start with a relatively high learning rate to make quick progress initially, and then decrease it as training progresses to fine-tune the model and avoid overshooting the optimal solution.

### Base Scheduler Class

I started by creating a base `scheduler` class to define the common interface:

```python
class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0 
        self.initial_lr = optimizer.init_lr # Store initial LR

    @abstractmethod
    def step(self):
        pass

    def get_lr(self):
        return self.optimizer.init_lr
```
This base class holds a reference to the optimizer and tracks the number of steps (which could be epochs or iterations depending on how it's used).

### StepLR (Already Provided)

The `StepLR` scheduler decreases the learning rate by a factor `gamma` every `step_size` steps. The update rule is:

$$ \text{lr}_{\text{new}} = \text{lr}_{\text{old}} \times \gamma \quad \text{if} \quad \text{step\_count} \mod \text{step\_size} == 0 $$

### MultiStepLR

This scheduler is similar to `StepLR`, but instead of decaying the learning rate at regular intervals, it decays it at specific milestones (e.g., specific epoch numbers).

Let $\text{lr}_0$ be the initial learning rate, $M = \{m_1, m_2, ..., m_k\}$ be the set of milestone steps (epochs), and $\gamma$ be the decay factor. The learning rate $\text{lr}_t$ at step $t$ is updated as follows:

$$ \text{lr}_t = \text{lr}_0 \times \gamma^k \quad \text{where } k = |\{m \in M \mid m \le t\}| $$

In simpler terms, every time the current step $t$ crosses a milestone $m_i$, the learning rate is multiplied by $\gamma$.

My implementation looks like this:

```python
class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        # ... (error checking for milestones)
        self.milestones = milestones
        self.gamma = gamma
        self.last_lr_update_step = -1 

    def step(self) -> None:
        self.step_count += 1
        if self.step_count in self.milestones and self.step_count > self.last_lr_update_step:
            new_lr = self.optimizer.init_lr * self.gamma
            # ... (optional print statement)
            self.optimizer.init_lr = new_lr
            self.last_lr_update_step = self.step_count 
```
This allows for more flexible control over when the learning rate changes compared to `StepLR`.

### ExponentialLR

The `ExponentialLR` scheduler decays the learning rate by a factor `gamma` at *every* step.

The update rule is:

$$ \text{lr}_t = \text{lr}_{t-1} \times \gamma $$

Or, expressed in terms of the initial learning rate $\text{lr}_0$:

$$ \text{lr}_t = \text{lr}_0 \times \gamma^t $$

where $t$ is the current step number.

The implementation is quite straightforward:

```python
class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        new_lr = self.optimizer.init_lr * self.gamma
        self.optimizer.init_lr = new_lr
```
This scheduler provides a smooth, continuous decay of the learning rate over time.

Implementing these schedulers helped me understand how dynamically adjusting hyperparameters like the learning rate can be beneficial during the training process. Choosing the right scheduler and its parameters (like milestones or gamma) often requires experimentation and depends on the specific dataset and model architecture.

## Momentum Gradient Descent (MomentGD)

While standard Stochastic Gradient Descent (SGD) is a good starting point, it can sometimes be slow to converge, especially in areas with high curvature or noisy gradients. I learned about Momentum Gradient Descent (MomentGD) as a way to improve upon SGD.

### The Problem with SGD

Imagine a ball rolling down a hilly landscape (representing the loss surface). SGD takes steps directly in the direction of the steepest descent at each point. If the landscape has narrow valleys or ravines, SGD can oscillate back and forth across the valley instead of moving smoothly along the bottom towards the minimum.

### Introducing Momentum

Momentum helps address this by adding a "velocity" term to the update rule. Think of it like giving the rolling ball some inertia. Instead of just considering the current gradient, the update also incorporates a fraction of the previous update direction.

The update rules for Momentum GD are:

1.  **Update Velocity:**
    $$ v_t = \mu v_{t-1} - \eta \nabla L(\theta_{t-1}) $$

2.  **Update Parameters:**
    $$ \theta_t = \theta_{t-1} + v_t $$

where:
- $\theta$ represents the parameters (weights and biases)
- $\eta$ is the learning rate (`init_lr` in my code)
- $\nabla L(\theta_{t-1})$ is the gradient of the loss function with respect to the parameters at the previous step (`layer.grads[key]` in my code, potentially including weight decay)
- $\mu$ is the momentum coefficient (usually around 0.9)
- $v_t$ is the velocity (or momentum) vector at step $t$

The velocity $v_t$ is essentially an exponentially decaying moving average of the past gradients. If gradients consistently point in the same direction, the velocity builds up, leading to faster convergence. If gradients oscillate, the momentum term helps to dampen these oscillations.

### Implementation Details

In my implementation, I needed to store the velocity for each parameter. I used a dictionary `self.velocities` for this:

```python
class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        # Initialize velocity dictionary
        self.velocities = {} 
        for i, layer in enumerate(self.model.layers):
             if hasattr(layer, 'optimizable') and layer.optimizable:
                self.velocities[i] = {}
                for key in layer.params.keys():
                    # Initialize velocity to zero
                    self.velocities[i][key] = np.zeros_like(layer.params[key])

    def step(self):
        for i, layer in enumerate(self.model.layers):
             if hasattr(layer, 'optimizable') and layer.optimizable:
                for key in layer.params.keys():
                    # Calculate gradient (including weight decay if needed)
                    grad_update = layer.grads[key]
                    if hasattr(layer, 'weight_decay') and layer.weight_decay:
                        grad_update += layer.weight_decay_lambda * layer.params[key]

                    # Update velocity: v = mu * v - lr * grad
                    self.velocities[i][key] = self.mu * self.velocities[i][key] - self.init_lr * grad_update

                    # Update parameter: param = param + v
                    layer.params[key] += self.velocities[i][key]
```

Implementing Momentum GD was interesting because it showed me how a simple modification to the basic SGD update rule could lead to significantly better optimization behavior in practice. It helps the optimizer "remember" past directions and move more confidently towards the minimum.
