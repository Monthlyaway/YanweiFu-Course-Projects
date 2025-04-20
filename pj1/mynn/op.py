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
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # Store input for backward pass
        self.input = X
        # Linear transformation: Y = X·W + b
        # X: [batch_size, in_dim], W: [in_dim, out_dim], b: [1, out_dim]
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = self.input.shape[0]
        
        # Gradient with respect to W: dL/dW = X^T · dL/dY
        # self.input: [batch_size, in_dim], grad: [batch_size, out_dim]
        # dW: [in_dim, out_dim]
        dW = np.dot(self.input.T, grad)
        
        # Gradient with respect to b: dL/db = sum(dL/dY, axis=0)
        # Sum gradients across the batch dimension
        # db: [1, out_dim]
        db = np.sum(grad, axis=0, keepdims=True)
        
        # Gradient with respect to X: dL/dX = dL/dY · W^T
        # grad: [batch_size, out_dim], W.T: [out_dim, in_dim]
        # dX: [batch_size, in_dim]
        dX = np.dot(grad, self.W.T)
        
        # Store gradients for optimizer
        self.grads['W'] = dW
        self.grads['b'] = db
        
        # Apply weight decay if enabled
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
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
            raise NotImplementedError("Backward without softmax not implemented")
            
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