from abc import abstractmethod
import numpy as np


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """

    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None  # Record the input for backward process.

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay  # whether using weight decay
        # control the intensity of weight decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # Store input for backward pass
        self.input = X
        print(f"[Linear Forward] Input shape: {X.shape}, W shape: {self.W.shape}, b shape: {self.b.shape}")
        # Linear transformation: Y = X·W + b
        # X: [batch_size, in_dim], W: [in_dim, out_dim], b: [1, out_dim]
        output = np.dot(X, self.W) + self.b
        print(f"[Linear Forward] Output shape: {output.shape}")
        return output

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        print(f"[Linear Backward] Input gradient shape: {grad.shape}, stored input shape: {self.input.shape}")
        batch_size = self.input.shape[0]

        # Gradient with respect to W: dL/dW = X^T · dL/dY
        # self.input: [batch_size, in_dim], grad: [batch_size, out_dim]
        # dW: [in_dim, out_dim]
        dW = np.dot(self.input.T, grad)
        print(f"[Linear Backward] dW shape: {dW.shape}")

        # Gradient with respect to b: dL/db = sum(dL/dY, axis=0)
        # Sum gradients across the batch dimension
        # db: [1, out_dim]
        db = np.sum(grad, axis=0, keepdims=True)
        print(f"[Linear Backward] db shape: {db.shape}")

        # Gradient with respect to X: dL/dX = dL/dY · W^T
        # grad: [batch_size, out_dim], W.T: [out_dim, in_dim]
        # dX: [batch_size, in_dim]
        dX = np.dot(grad, self.W.T)
        print(f"[Linear Backward] dX shape: {dX.shape}")

        # Store gradients for optimizer
        self.grads['W'] = dW
        self.grads['b'] = db

        # Apply weight decay if enabled
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        # Handle scalar or tuple kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # Handle scalar or tuple stride
        if isinstance(stride, int):
            stride = (stride, stride)

        # Handle scalar or tuple padding
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        # W shape: [out_channels, in_channels, kernel_height, kernel_width]
        self.W = initialize_method(
            size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # Bias shape: [out_channels, 1, 1]
        self.b = initialize_method(size=(out_channels, 1, 1))

        # Store parameters and gradients
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

        # Store inputs for backward pass
        self.input = None
        self.input_padded = None

        # Weight decay parameters
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, kernel_height, kernel_width]
        output: [batch, out_channels, output_height, output_width]
        """
        # Store input for backward pass
        self.input = X
        print(f"[Conv2D Forward] Input shape: {X.shape}, W shape: {self.W.shape}, b shape: {self.b.shape}")

        # Get dimensions
        batch_size, _, input_height, input_width = X.shape

        # Apply padding if specified
        if self.padding[0] > 0 or self.padding[1] > 0:
            # Create padded input
            pad_height, pad_width = self.padding
            input_padded = np.zeros((batch_size, self.in_channels,
                                     input_height + 2 * pad_height,
                                     input_width + 2 * pad_width))

            # Copy original input to the center of padded input
            input_padded[:, :, pad_height:pad_height + input_height,
                         pad_width:pad_width + input_width] = X

            # Store padded input for backward pass
            self.input_padded = input_padded
        else:
            # No padding
            self.input_padded = X

        # Calculate output dimensions
        padded_height = input_height + 2 * self.padding[0]
        padded_width = input_width + 2 * self.padding[1]
        output_height = (
            padded_height - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (
            padded_width - self.kernel_size[1]) // self.stride[1] + 1

        # Initialize output
        output = np.zeros((batch_size, self.out_channels,
                          output_height, output_width))
        print(f"[Conv2D Forward] Output shape: {output.shape}")

        # Perform convolution operation
        for b in range(batch_size):  # Batch dimension
            for oc in range(self.out_channels):  # Output channel dimension
                for h in range(output_height):
                    h_start = h * self.stride[0]
                    h_end = h_start + self.kernel_size[0]

                    for w in range(output_width):
                        w_start = w * self.stride[1]
                        w_end = w_start + self.kernel_size[1]

                        # Extract the current patch from the input
                        patch = self.input_padded[b, :,
                                                  h_start:h_end, w_start:w_end]

                        # Perform convolution (element-wise multiplication and sum)
                        # Shape: [in_channels, kernel_height, kernel_width]
                        output[b, oc, h, w] = np.sum(
                            patch * self.W[oc]) + self.b[oc, 0, 0]

        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, output_height, output_width]
        return: [batch_size, in_channels, input_height, input_width]
        """
        print(f"[Conv2D Backward] Input gradient shape: {grads.shape}, stored input shape: {self.input.shape}")
        # Get dimensions
        batch_size, _, output_height, output_width = grads.shape
        _, _, input_height, input_width = self.input.shape

        # Initialize gradients for weights, bias, and input
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(self.input_padded)
        print(f"[Conv2D Backward] dW shape: {dW.shape}, db shape: {db.shape}, dX_padded shape: {dX_padded.shape}")


        # Calculate gradients
        for b in range(batch_size):  # Batch dimension
            for oc in range(self.out_channels):  # Output channel dimension
                for h in range(output_height):
                    h_start = h * self.stride[0]
                    h_end = h_start + self.kernel_size[0]

                    for w in range(output_width):
                        w_start = w * self.stride[1]
                        w_end = w_start + self.kernel_size[1]

                        # Extract the current patch from the input
                        patch = self.input_padded[b, :,
                                                  h_start:h_end, w_start:w_end]

                        # Calculate gradient for weights
                        # Element-wise multiplication of patch with the gradient for this output
                        dW[oc] += patch * grads[b, oc, h, w]

                        # Calculate gradient for bias (just the gradient itself)
                        db[oc, 0, 0] += grads[b, oc, h, w]

                        # Calculate gradient for input (backpropagation to input)
                        # Full convolution using the weights and the gradient for this output
                        dX_padded[b, :, h_start:h_end,
                                  w_start:w_end] += self.W[oc] * grads[b, oc, h, w]

        # Store gradients for the optimizer
        self.grads['W'] = dW
        self.grads['b'] = db

        # Apply weight decay if enabled
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        # Remove padding from dX_padded to get dX if padding was applied
        if self.padding[0] > 0 or self.padding[1] > 0:
            pad_height, pad_width = self.padding
            dX = dX_padded[:, :, pad_height:pad_height + input_height,
                           pad_width:pad_width + input_width]
        else:
            dX = dX_padded

        print(f"[Conv2D Backward] Final dX shape: {dX.shape}")
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class ReLU(Layer):
    """
    An activation layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        print(f"[ReLU Forward] Input shape: {X.shape}")
        self.input = X
        output = np.where(X < 0, 0, X)
        print(f"[ReLU Forward] Output shape: {output.shape}")
        return output

    def backward(self, grads):
        print(f"[ReLU Backward] Input gradient shape: {grads.shape}, stored input shape: {self.input.shape}")
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        print(f"[ReLU Backward] Output gradient shape: {output.shape}")
        return output


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """

    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  # By default, softmax is applied
        self.input = None        # Store input for backward pass
        self.softmax_output = None  # Store softmax output
        self.labels = None       # Store labels for backward pass
        self.optimizable = False
        self.grads = None        # Store gradients for backward pass

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.input = predicts
        self.labels = labels
        batch_size = predicts.shape[0]

        # Apply softmax if required
        if self.has_softmax:
            self.softmax_output = softmax(predicts)
        else:
            self.softmax_output = predicts

        # Create one-hot encoded labels
        # Note: This could be optimized in a real implementation
        one_hot_labels = np.zeros((batch_size, self.max_classes))
        for i in range(batch_size):
            one_hot_labels[i, labels[i]] = 1

        # Calculate cross-entropy loss
        # L = -sum(y_true * log(y_pred))
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        log_probs = np.log(self.softmax_output + epsilon)
        loss = -np.sum(one_hot_labels * log_probs) / batch_size

        return loss

    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.input.shape[0]

        # If softmax was applied in the forward pass, gradient is (softmax_output - one_hot_labels)
        # Otherwise, gradient needs to be computed based on specific loss function
        if self.has_softmax:
            # Create one-hot encoded labels
            one_hot_labels = np.zeros((batch_size, self.max_classes))
            for i in range(batch_size):
                one_hot_labels[i, self.labels[i]] = 1

            # Gradient of softmax cross-entropy: softmax_output - one_hot_labels
            # Divide by batch_size for averaging
            self.grads = (self.softmax_output - one_hot_labels) / batch_size
        else:
            # If no softmax was applied, this would depend on the loss function used
            # For simple cross-entropy with probabilities, this might be different
            raise NotImplementedError(
                "Backward without softmax not implemented")

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self


class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


class CrossEntropyLoss(Layer):
    """
    Implements cross entropy loss with softmax activation for multi-class classification.
    
    The softmax function converts raw logits to probabilities using:
    p(y_i) = exp(z_i) / sum(exp(z_j))
    
    The cross entropy loss then computes:
    L = -sum(y_true * log(p(y_i)))
    
    Where:
    - z_i are the raw logits from the network
    - y_true are one-hot encoded ground truth labels
    """
    
    def __init__(self, model=None, num_classes=10) -> None:
        """
        Initialize the CrossEntropyLoss layer.
        
        Args:
            model: The model this loss is attached to (for backward propagation)
            num_classes: Number of classes in the classification task
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.input = None        # Store input logits for backward pass
        self.softmax_output = None  # Store softmax probabilities
        self.labels = None       # Store true labels for backward pass
        self.optimizable = False
        self.grads = None        # Store gradients for backward pass
    
    def __call__(self, logits, labels):
        """
        Compute the cross entropy loss for the given logits and labels.
        
        Args:
            logits: Raw output from the network [batch_size, num_classes]
            labels: Ground truth labels [batch_size,]
            
        Returns:
            Cross entropy loss value (scalar)
        """
        return self.forward(logits, labels)
    
    def forward(self, logits, labels):
        """
        Forward pass to compute the cross entropy loss.
        
        Args:
            logits: Raw output from the network [batch_size, num_classes]
            labels: Ground truth labels [batch_size,]
            
        Returns:
            Cross entropy loss value (scalar)
        """
        self.input = logits
        self.labels = labels
        batch_size = logits.shape[0]
        
        # Apply softmax with numerical stability
        # Subtract max value for numerical stability
        self.softmax_output = softmax(logits)
        
        # Create one-hot encoded labels
        one_hot_labels = np.zeros((batch_size, self.num_classes))
        for i in range(batch_size):
            one_hot_labels[i, labels[i]] = 1
        
        # Calculate cross entropy loss
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        log_probs = np.log(self.softmax_output + epsilon)
        loss = -np.sum(one_hot_labels * log_probs) / batch_size
        
        return loss
    
    def backward(self):
        """
        Backward pass to compute gradients and propagate them through the model.
        
        The gradient of softmax cross entropy loss with respect to logits is:
        dL/dz_i = p(y_i) - y_true_i
        
        Returns:
            None (gradients are stored and passed to the model)
        """
        batch_size = self.input.shape[0]
        
        # Create one-hot encoded labels
        one_hot_labels = np.zeros((batch_size, self.num_classes))
        for i in range(batch_size):
            one_hot_labels[i, self.labels[i]] = 1
        
        # Compute gradient: softmax_output - one_hot_labels
        # Normalize by batch size
        self.grads = (self.softmax_output - one_hot_labels) / batch_size
        
        # Propagate gradients through the model
        if self.model is not None:
            self.model.backward(self.grads)
        
        return self.grads
