from .op import *
import pickle


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """

    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_path):
        """
        Load model parameters from a file
        param_path: Path to the pickle file containing model parameters
        """
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        # Initialize the layers list
        self.layers = []

        # Create layers based on loaded parameters
        for i in range(len(self.size_list) - 1):
            # Create linear layer
            layer = Linear(
                in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']

            # Add linear layer
            self.layers.append(layer)

            # Add activation layer if not the last linear layer
            if i < len(self.size_list) - 2:
                if self.act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                    self.layers.append(layer_f)

    def save_model(self, save_path):
        """
        Save model parameters to a file
        save_path: Path where to save the pickle file
        """
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """

    def __init__(self, channels_list=None, kernel_sizes=None, strides=None, paddings=None,
                 fc_sizes=None, act_func=None, weight_decay_lambdas=None):
        super().__init__()

        # Store configuration parameters
        # List of channel sizes [in_channels, conv1_out, conv2_out, ...]
        self.channels_list = channels_list
        self.kernel_sizes = kernel_sizes    # List of kernel sizes for each conv layer
        self.strides = strides              # List of strides for each conv layer
        self.paddings = paddings            # List of paddings for each conv layer
        self.fc_sizes = fc_sizes            # List of fully connected layer sizes
        # Activation function (e.g., 'ReLU')
        self.act_func = act_func
        self.layers = []                    # Will hold all layers in the network

        # Only create the network if all parameters are provided
        if (channels_list is not None and kernel_sizes is not None and
            strides is not None and paddings is not None and
                fc_sizes is not None and act_func is not None):

            # Create convolutional layers
            for i in range(len(channels_list) - 1):
                # Add convolutional layer
                conv_layer = conv2D(
                    in_channels=channels_list[i],
                    out_channels=channels_list[i+1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i]
                )

                # Apply weight decay if specified
                if weight_decay_lambdas is not None:
                    conv_layer.weight_decay = True
                    conv_layer.weight_decay_lambda = weight_decay_lambdas[i]

                self.layers.append(conv_layer)

                # Add activation function after each conv layer
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")

            # Create fully connected layers
            # Last FC layer doesn't have activation
            for i in range(len(fc_sizes) - 1):
                fc_layer = Linear(in_dim=fc_sizes[i], out_dim=fc_sizes[i+1])

                # Apply weight decay if specified
                if weight_decay_lambdas is not None:
                    fc_layer.weight_decay = True
                    # Use the last weight decay lambda for FC layers
                    fc_layer.weight_decay_lambda = weight_decay_lambdas[-1]

                self.layers.append(fc_layer)

                # Add activation after each FC layer except the last one
                if i < len(fc_sizes) - 2:
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    elif act_func == 'Logistic':
                        raise NotImplementedError(
                            "Logistic activation not implemented")

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass through the CNN network
        X: Input with shape [batch_size, in_channels, height, width]
        Returns: Output of the network
        """
        assert self.layers, 'Model has not been initialized yet. Use model.load_model to load a model or create a new model with all required parameters.'

        outputs = X

        # Need to reshape before FC layers (from conv output to FC input)
        reshape_needed = True
        for i, layer in enumerate(self.layers):
            # Check if we're transitioning from conv to fc layers
            if reshape_needed and isinstance(layer, Linear):
                # Reshape from [batch_size, channels, height, width] to [batch_size, flat_dim]
                batch_size = outputs.shape[0]
                outputs = outputs.reshape(batch_size, -1)
                reshape_needed = False

            outputs = layer(outputs)

        return outputs

    def backward(self, loss_grad):
        """
        Backward pass through the CNN network
        loss_grad: Gradient from the loss function with respect to the network output
        Returns: Gradient with respect to the input
        """
        grads = loss_grad

        # Need to reshape gradients when transitioning between fc and conv layers
        need_reshape = False
        conv_shape = None

        # Iterate through layers in reverse order
        for i, layer in enumerate(reversed(self.layers)):
            # If we're at a Linear layer and the next layer (going backward) is a conv/ReLU,
            # we'll need to reshape
            if isinstance(layer, Linear) and i < len(self.layers) - 1:
                next_layer = list(reversed(self.layers))[i+1]
                if isinstance(next_layer, conv2D) or (isinstance(next_layer, ReLU) and
                                                      i+2 < len(self.layers) and
                                                      isinstance(list(reversed(self.layers))[i+2], conv2D)):
                    need_reshape = True
                    # Store the shape from the conv layer for reshaping
                    if isinstance(next_layer, conv2D):
                        conv_shape = next_layer.input.shape
                    else:  # It's a ReLU after a conv
                        conv_shape = list(reversed(self.layers))[
                            i+2].input.shape

            # Perform the backward pass through this layer
            grads = layer.backward(grads)

            # Reshape if needed after the backward pass
            if need_reshape and not (isinstance(layer, Linear)):
                # We've passed through the Linear layer, reshape back to conv format
                batch_size = grads.shape[0]
                # Use the stored conv shape to reshape correctly
                grads = grads.reshape(conv_shape)
                need_reshape = False

        return grads

    def load_model(self, param_path):
        """
        Load model parameters from a file
        param_path: Path to the pickle file containing model parameters
        """
        with open(param_path, 'rb') as f:
            params = pickle.load(f)

        # Extract configuration parameters
        self.channels_list = params[0]
        self.kernel_sizes = params[1]
        self.strides = params[2]
        self.paddings = params[3]
        self.fc_sizes = params[4]
        self.act_func = params[5]

        # Recreate the layers
        self.layers = []

        # Parameter index counter
        param_idx = 6

        # Create convolutional layers
        for i in range(len(self.channels_list) - 1):
            # Add convolutional layer
            conv_layer = conv2D(
                in_channels=self.channels_list[i],
                out_channels=self.channels_list[i+1],
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                padding=self.paddings[i]
            )

            # Load parameters
            conv_layer.W = params[param_idx]['W']
            conv_layer.b = params[param_idx]['b']
            conv_layer.params['W'] = conv_layer.W
            conv_layer.params['b'] = conv_layer.b
            conv_layer.weight_decay = params[param_idx]['weight_decay']
            conv_layer.weight_decay_lambda = params[param_idx]['lambda']
            param_idx += 1

            self.layers.append(conv_layer)

            # Add activation function after each conv layer
            if self.act_func == 'ReLU':
                self.layers.append(ReLU())
            elif self.act_func == 'Logistic':
                raise NotImplementedError(
                    "Logistic activation not implemented")

        # Create fully connected layers
        for i in range(len(self.fc_sizes) - 1):
            fc_layer = Linear(
                in_dim=self.fc_sizes[i], out_dim=self.fc_sizes[i+1])

            # Load parameters
            fc_layer.W = params[param_idx]['W']
            fc_layer.b = params[param_idx]['b']
            fc_layer.params['W'] = fc_layer.W
            fc_layer.params['b'] = fc_layer.b
            fc_layer.weight_decay = params[param_idx]['weight_decay']
            fc_layer.weight_decay_lambda = params[param_idx]['lambda']
            param_idx += 1

            self.layers.append(fc_layer)

            # Add activation after each FC layer except the last one
            if i < len(self.fc_sizes) - 2:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.act_func == 'Logistic':
                    raise NotImplementedError(
                        "Logistic activation not implemented")

    def save_model(self, save_path):
        """
        Save model parameters to a file
        save_path: Path where to save the pickle file
        """
        # Save configuration parameters
        params = [
            self.channels_list,
            self.kernel_sizes,
            self.strides,
            self.paddings,
            self.fc_sizes,
            self.act_func
        ]

        # Save layer parameters
        for layer in self.layers:
            if layer.optimizable:
                params.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        # Write to file
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
