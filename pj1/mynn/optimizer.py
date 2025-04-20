from abc import abstractmethod
import numpy as np


class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, init_lr, model) -> None:
        """
        Initializes the optimizer.

        Args:
            init_lr (float): The initial learning rate.
            model: The model whose parameters will be optimized.
        """
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        """Performs a single optimization step (parameter update)."""
        pass


class SGD(Optimizer):
    """Implements Stochastic Gradient Descent."""
    def __init__(self, init_lr, model):
        """
        Initializes the SGD optimizer.

        Args:
            init_lr (float): Learning rate.
            model: The model to optimize.
        """
        super().__init__(init_lr, model)

    def step(self):
        """Performs a single optimization step using SGD."""
        for layer in self.model.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                for key in layer.params.keys():
                    # Apply weight decay if enabled for the layer
                    # Note: Weight decay is often applied *before* the gradient step in SGD.
                    # Here it's combined, which is also common.
                    # layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda) # Original placement
                    
                    # Calculate the gradient update
                    update = self.init_lr * layer.grads[key]

                    # Apply weight decay (L2 regularization) to the update if enabled
                    if hasattr(layer, 'weight_decay') and layer.weight_decay:
                         update += self.init_lr * layer.weight_decay_lambda * layer.params[key]

                    # Update the parameter
                    layer.params[key] -= update


class MomentGD(Optimizer):
    """
    Implements Momentum Gradient Descent.
    This optimizer helps accelerate SGD in the relevant direction and dampens oscillations.
    """
    def __init__(self, init_lr, model, mu=0.9):
        """
        Initializes the Momentum Gradient Descent optimizer.

        Args:
            init_lr (float): Learning rate.
            model: The model to optimize.
            mu (float): Momentum factor (typically close to 1, e.g., 0.9). Default: 0.9.
        """
        super().__init__(init_lr, model)
        self.mu = mu
        # Initialize velocity dictionary to store momentum for each parameter
        self.velocities = {}
        for i, layer in enumerate(self.model.layers):
             if hasattr(layer, 'optimizable') and layer.optimizable:
                self.velocities[i] = {}
                for key in layer.params.keys():
                    # Initialize velocity to zero with the same shape as the parameter
                    self.velocities[i][key] = np.zeros_like(layer.params[key])

    def step(self):
        """Performs a single optimization step using Momentum GD."""
        for i, layer in enumerate(self.model.layers):
             if hasattr(layer, 'optimizable') and layer.optimizable:
                for key in layer.params.keys():
                    # Calculate the raw gradient update (including weight decay if applicable)
                    grad_update = layer.grads[key]
                    if hasattr(layer, 'weight_decay') and layer.weight_decay:
                        grad_update += layer.weight_decay_lambda * layer.params[key]

                    # Update velocity: v = mu * v - lr * grad
                    self.velocities[i][key] = self.mu * self.velocities[i][key] - self.init_lr * grad_update

                    # Update parameter: param = param + v
                    layer.params[key] += self.velocities[i][key]