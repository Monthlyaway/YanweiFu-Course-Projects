from abc import abstractmethod
import numpy as np

class scheduler():
    """Base class for learning rate schedulers."""
    def __init__(self, optimizer) -> None:
        """
        Initializes the scheduler.

        Args:
            optimizer: The optimizer instance whose learning rate will be adjusted.
        """
        self.optimizer = optimizer
        self.step_count = 0 # Tracks the number of steps taken (usually epochs or iterations)
        # Store the initial learning rate from the optimizer
        self.initial_lr = optimizer.init_lr

    @abstractmethod
    def step(self):
        """Performs a scheduler step, potentially adjusting the learning rate."""
        pass

    def get_lr(self):
        """Returns the current learning rate of the optimizer."""
        return self.optimizer.init_lr


class StepLR(scheduler):
    """
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    """
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        """
        Args:
            optimizer: The optimizer instance.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        """
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        """Updates the learning rate based on the step count."""
        self.step_count += 1
        # Decay learning rate only when step_count reaches step_size
        if self.step_count % self.step_size == 0:
            new_lr = self.optimizer.init_lr * self.gamma
            print(f"StepLR: Epoch {self.step_count}, reducing learning rate from {self.optimizer.init_lr:.6f} to {new_lr:.6f}")
            self.optimizer.init_lr = new_lr


class MultiStepLR(scheduler):
    """
    Decays the learning rate of each parameter group by gamma once the number of
    steps reaches one of the milestones.
    """
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        """
        Args:
            optimizer: The optimizer instance.
            milestones (list): List of step indices (e.g., epochs). Must be increasing.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        """
        super().__init__(optimizer)
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of increasing integers.')
        self.milestones = milestones
        self.gamma = gamma
        self.last_lr_update_step = -1 # Track the step of the last LR update

    def step(self) -> None:
        """Updates the learning rate if the current step is a milestone."""
        self.step_count += 1
        # Check if the current step is a milestone and if we haven't already updated at this step
        if self.step_count in self.milestones and self.step_count > self.last_lr_update_step:
            new_lr = self.optimizer.init_lr * self.gamma
            print(f"MultiStepLR: Step {self.step_count}, reducing learning rate from {self.optimizer.init_lr:.6f} to {new_lr:.6f}")
            self.optimizer.init_lr = new_lr
            self.last_lr_update_step = self.step_count # Record this step as the last update


class ExponentialLR(scheduler):
    """
    Decays the learning rate of each parameter group by gamma every step.
    """
    def __init__(self, optimizer, gamma) -> None:
        """
        Args:
            optimizer: The optimizer instance.
            gamma (float): Multiplicative factor of learning rate decay.
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        """Updates the learning rate exponentially."""
        self.step_count += 1
        # Decay learning rate at every step
        new_lr = self.optimizer.init_lr * self.gamma
        # Optional: Add a print statement for debugging, might be too verbose
        # print(f"ExponentialLR: Step {self.step_count}, reducing learning rate from {self.optimizer.init_lr:.6f} to {new_lr:.6f}")
        self.optimizer.init_lr = new_lr