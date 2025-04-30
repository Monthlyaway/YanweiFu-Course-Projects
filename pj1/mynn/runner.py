import numpy as np
import os
from tqdm import tqdm  # Optional: for progress bars


class RunnerM():
    """
    Manages the training and evaluation process for a neural network model.

    This class orchestrates the training loop, including data handling,
    forward and backward passes, optimization, learning rate scheduling,
    evaluation, and model saving/loading.
    """

    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None, 
                 is_cnn=False, preprocess_fn=None):
        """
        Initializes the Runner.

        Args:
            model: The neural network model instance (e.g., Model_MLP, Model_CNN).
                   Expected to have __call__, backward, save_model, load_model methods.
            optimizer: The optimizer instance (e.g., SGD, MomentGD).
                       Expected to have a step() method.
            metric: A function to compute the evaluation metric (e.g., accuracy).
                    Expected signature: metric(predictions, labels) -> score.
            loss_fn: The loss function instance (e.g., MultiCrossEntropyLoss).
                     Expected signature: loss_fn(predictions, labels) -> loss_value.
                     Expected to have a backward() method that triggers model backpropagation.
            batch_size (int): The number of samples per batch during training. Default: 32.
            scheduler: An optional learning rate scheduler instance.
                       Expected to have a step() method. Default: None.
            is_cnn (bool): Flag indicating if the model is a CNN. This affects data handling.
                           Default: False.
            preprocess_fn: Optional function to preprocess data before feeding to model.
                           Signature: preprocess_fn(X) -> processed_X. Default: None.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.is_cnn = is_cnn
        self.preprocess_fn = preprocess_fn

        # Lists to store training history (recorded per epoch)
        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self.best_score = -np.inf  # Initialize best score for model saving

    def _preprocess_batch(self, X):
        """
        Preprocess a batch of data before feeding to the model.
        
        Args:
            X: Input data batch
            
        Returns:
            Preprocessed data ready for the model
        """
        # Apply custom preprocessing if provided
        if self.preprocess_fn is not None:
            X = self.preprocess_fn(X)
            
        # Handle CNN input formatting if needed
        if self.is_cnn and len(X.shape) == 3:  # If shape is [batch_size, height, width]
            # Add channel dimension for grayscale images: [batch_size, height, width] -> [batch_size, 1, height, width]
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        
        return X

    def train(self, train_set, dev_set, **kwargs):
        """
        Trains the model for a specified number of epochs.

        Args:
            train_set (tuple): A tuple (X_train, y_train) containing training data and labels.
            dev_set (tuple): A tuple (X_dev, y_dev) containing development/validation data and labels.
            **kwargs: Additional training parameters:
                num_epochs (int): Number of epochs to train for. Default: 0.
                log_iters (int): Print training progress every `log_iters` iterations. Default: 100.
                save_dir (str): Directory to save the best performing model. Default: "best_model".
                scheduler_per_epoch (bool): Whether to step the scheduler once per epoch instead of 
                                           per iteration. Default: False.
                early_stopping (int): Number of epochs to wait for improvement before early stopping.
                                     Set to 0 to disable early stopping. Default: 0.
        """
        # --- Training Setup ---
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        scheduler_per_epoch = kwargs.get("scheduler_per_epoch", False)
        early_stopping = kwargs.get("early_stopping", 0)

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            # Use makedirs to create parent dirs if needed
            os.makedirs(save_dir)

        print(f"Starting training for {num_epochs} epochs...")
        
        # Early stopping variables
        epochs_without_improvement = 0
        
        # --- Epoch Loop ---
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            X_train, y_train = train_set
            assert X_train.shape[0] == y_train.shape[0], "Training data and labels must have the same number of samples."

            # Shuffle training data at the beginning of each epoch
            idx = np.random.permutation(range(X_train.shape[0]))
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]

            epoch_train_loss_list = []  # Temporary list for batch losses within epoch
            epoch_train_scores_list = []  # Temporary list for batch scores within epoch

            # --- Iteration (Batch) Loop ---
            num_batches = int(np.ceil(X_train.shape[0] / self.batch_size))
            
            # Use tqdm if available for progress bar
            iterator = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}") if 'tqdm' in globals() else range(num_batches)
            
            for iteration in iterator:
                # Get current batch
                start_idx = iteration * self.batch_size
                end_idx = min((iteration + 1) *
                              self.batch_size, X_train.shape[0])
                batch_X = X_train_shuffled[start_idx:end_idx]
                batch_y = y_train_shuffled[start_idx:end_idx]

                if batch_X.shape[0] == 0:  # Skip empty batches
                    continue

                # Preprocess data if needed (especially important for CNNs)
                batch_X = self._preprocess_batch(batch_X)

                # --- Core Training Steps ---
                # 1. Forward Pass: Compute model predictions (logits)
                logits = self.model(batch_X)

                # 2. Calculate Loss
                trn_loss = self.loss_fn(logits, batch_y)
                epoch_train_loss_list.append(trn_loss)

                # 3. Calculate Training Metric (e.g., accuracy) for the batch
                trn_score = self.metric(logits, batch_y)
                epoch_train_scores_list.append(trn_score)

                # 4. Backward Pass: Compute gradients (triggered by loss function's backward method)
                self.loss_fn.backward()  # Assumes loss_fn calls model.backward() internally

                # 5. Optimizer Step: Update model parameters based on gradients
                self.optimizer.step()

                # 6. Scheduler Step: Adjust learning rate (if scheduler is used and per-iteration stepping is enabled)
                if self.scheduler is not None and not scheduler_per_epoch:
                    self.scheduler.step()

                # --- Logging Iteration Progress (Optional) ---
                if (iteration + 1) % log_iters == 0:
                    # Log average loss/score over the last log_iters batches
                    avg_batch_loss = np.mean(
                        epoch_train_loss_list[-log_iters:])
                    avg_batch_score = np.mean(
                        epoch_train_scores_list[-log_iters:])
                    log_msg = f"Iter {iteration+1}/{num_batches} | Loss: {avg_batch_loss:.4f}, Score: {avg_batch_score:.4f}"
                    
                    # Update tqdm description if using tqdm, otherwise print
                    if 'tqdm' in globals():
                        iterator.set_postfix_str(log_msg)
                    else:
                        print(f"  {log_msg}")

            # --- End of Epoch ---
            # Calculate average training loss and score for the completed epoch
            avg_epoch_train_loss = np.mean(epoch_train_loss_list)
            avg_epoch_train_score = np.mean(epoch_train_scores_list)
            self.train_loss.append(avg_epoch_train_loss)
            self.train_scores.append(avg_epoch_train_score)

            # Step scheduler once per epoch if configured that way
            if self.scheduler is not None and scheduler_per_epoch:
                self.scheduler.step()

            # Evaluate on the full development set at the end of the epoch
            dev_score, dev_loss = self.evaluate(dev_set)
            # Append epoch-level development metrics
            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)

            print(
                f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_epoch_train_loss:.4f}, Avg Train Score: {avg_epoch_train_score:.4f}")
            print(
                f"                 Dev Loss: {dev_loss:.4f}, Dev Score: {dev_score:.4f}")
            if self.scheduler:
                print(
                    f"                 Current LR: {self.scheduler.get_lr():.6f}")

            # --- Model Checkpointing ---
            # Save the model if it achieves the best score on the development set so far
            if dev_score > self.best_score:
                old_best_score = self.best_score
                self.best_score = dev_score
                self.best_epoch = epoch + 1  # 更新最佳epoch
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(
                    f"  ** Best dev score improved: {old_best_score:.5f} --> {self.best_score:.5f}. Model saved to {save_path} **")
                # Reset early stopping counter
                epochs_without_improvement = 0
            else:
                # Increment early stopping counter
                epochs_without_improvement += 1
                if early_stopping > 0 and epochs_without_improvement >= early_stopping:
                    print(f"\nEarly stopping after {early_stopping} epochs without improvement.")
                    break

        print(
            f"\nTraining finished. Best development score: {self.best_score:.5f}")

    def evaluate(self, data_set):
        """
        Evaluates the model on a given dataset.

        Args:
            data_set (tuple): A tuple (X, y) containing data and labels.

        Returns:
            tuple: A tuple (score, loss) containing the evaluation metric score
                   and the average loss on the dataset.
        """
        X, y = data_set
        if X.shape[0] == 0:
            return 0.0, 0.0  # Handle empty dataset
            
        # Preprocess the evaluation data if needed
        X = self._preprocess_batch(X)
        
        # For large datasets, we might need to evaluate in batches
        if X.shape[0] > 1000 and self.is_cnn:  # Arbitrary threshold for batch evaluation
            return self._batch_evaluate(X, y)

        # Perform forward pass on the entire dataset
        logits = self.model(X)

        # Calculate loss
        loss = self.loss_fn(logits, y)

        # Calculate evaluation metric (e.g., accuracy)
        score = self.metric(logits, y)

        return score, loss
        
    def _batch_evaluate(self, X, y):
        """
        Evaluates the model on a large dataset in batches to avoid memory issues.
        
        Args:
            X: Input data
            y: Target labels
            
        Returns:
            tuple: A tuple (avg_score, avg_loss) with average score and loss
        """
        total_loss = 0
        total_score = 0
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, X.shape[0])
            
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            
            # Forward pass
            logits = self.model(batch_X)
            
            # Compute loss and score
            batch_loss = self.loss_fn(logits, batch_y)
            batch_score = self.metric(logits, batch_y)
            
            # Weight by batch size for proper averaging
            batch_weight = len(batch_X) / len(X)
            total_loss += batch_loss * batch_weight
            total_score += batch_score * batch_weight
            
        return total_score, total_loss

    def save_model(self, save_path):
        """
        Saves the current model's parameters to a file.

        Args:
            save_path (str): The path where the model file should be saved.
        """
        print(f"Saving model parameters to {save_path}...")
        # Delegates saving to the model instance
        self.model.save_model(save_path)

    def load_model(self, load_path):
        """
        Loads model parameters from a saved file.

        Args:
            load_path (str): The path to the saved model file.
        """
        print(f"Loading model parameters from {load_path}...")
        # Delegates loading to the model instance
        self.model.load_model(load_path)
