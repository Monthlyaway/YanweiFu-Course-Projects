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
                        update += self.init_lr * \
                            layer.weight_decay_lambda * layer.params[key]

                    # Update the parameter
                    layer.params[key] -= update


class MomentGD(Optimizer):
    """
    Implements Momentum Gradient Descent.

    This optimizer adds a velocity term to SGD, which helps accelerate gradients
    in the right direction and dampens oscillations. It's especially helpful
    for navigating through ravines (areas where the curvature is much steeper in one
    dimension than in others).
    """

    def __init__(self, init_lr, model, mu=0.9):
        """
        Initializes the Momentum Gradient Descent optimizer.

        Args:
            init_lr (float): Initial learning rate.
            model: The model to optimize.
            mu (float): Momentum coefficient, controls the contribution
                        of the previous gradient step to the current iteration.
                        Typically set to a value between 0.5 and 0.99. Default: 0.9.
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
        """
        Performs a single optimization step using Momentum Gradient Descent.

        The update rule is:
        velocity = mu * velocity - learning_rate * gradient
        parameter = parameter + velocity
        """
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'optimizable') and layer.optimizable:
                for key in layer.params.keys():
                    # Calculate the gradient update (possibly including weight decay)
                    grad_update = layer.grads[key]

                    # Apply weight decay if the layer has it enabled
                    if hasattr(layer, 'weight_decay') and layer.weight_decay:
                        grad_update = grad_update + \
                            layer.weight_decay_lambda * layer.params[key]

                    # Update velocity: v = mu * v - lr * grad
                    self.velocities[i][key] = self.mu * \
                        self.velocities[i][key] - self.init_lr * grad_update

                    # Update parameter: param = param + v
                    layer.params[key] += self.velocities[i][key]


class AlternativeMomentGD(Optimizer):
    """
    Implements an alternative momentum-based gradient descent method.

    Uses the update rule:
    w^(t+1) = w^(t) - alpha_t * grad(f(w^t)) + beta_t * (w^(t) - w^(t-1))

    Where:
    - alpha_t is the learning rate (step size)
    - beta_t is the momentum strength (typically 0.9)
    - (w^(t) - w^(t-1)) is the difference between current and previous parameters

    This differs from traditional momentum which uses a velocity accumulation.
    """

    def __init__(self, init_lr, model, beta=0.9):
        """
        Initializes the Alternative Momentum Gradient Descent optimizer.

        Args:
            init_lr (float): Learning rate (alpha_t in the formula).
            model: The model to optimize.
            beta (float): Momentum coefficient (beta_t in the formula).
                          Default: 0.9.
        """
        super().__init__(init_lr, model)
        self.beta = beta

        # Initialize previous parameters dictionary to store w^(t-1)
        self.prev_params = {}

        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'optimizable') and layer.optimizable:
                self.prev_params[i] = {}
                for key in layer.params.keys():
                    # Initialize previous parameters with a copy of current parameters
                    self.prev_params[i][key] = layer.params[key].copy()

    def step(self):
        """
        Performs a single optimization step using the alternative momentum formula.

        The update rule is:
        w^(t+1) = w^(t) - alpha_t * grad(f(w^t)) + beta_t * (w^(t) - w^(t-1))
        """
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'optimizable') and layer.optimizable:
                for key in layer.params.keys():
                    # Current parameters (w^t)
                    current_params = layer.params[key].copy()

                    # Calculate the gradient update (possibly including weight decay)
                    grad_update = layer.grads[key]

                    # Apply weight decay if the layer has it enabled
                    if hasattr(layer, 'weight_decay') and layer.weight_decay:
                        grad_update = grad_update + \
                            layer.weight_decay_lambda * layer.params[key]

                    # Calculate parameter difference term: (w^(t) - w^(t-1))
                    param_diff = current_params - self.prev_params[i][key]

                    # Update parameters using the alternative momentum rule
                    layer.params[key] = current_params - \
                        self.init_lr * grad_update + self.beta * param_diff

                    # Store current parameters as previous for next iteration
                    self.prev_params[i][key] = current_params
