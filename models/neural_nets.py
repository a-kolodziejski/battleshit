import torch
import torch.nn as nn

class SimpleFCN(torch.nn.Module):
    '''
    Implements fully connected neural network built from linear layers.
    '''
    def __init__(self, input_dim, output_dim, hidden_dims,
                 hidden_activation, output_activation):
        '''
        Args:
            input_dim (int): dimensionality of input vector
            output_dim (int): dimensionality of output vector (= number of actions)
            hidden_dims (tuple of ints): dimensionality of hidden Linear layers
            hidden_activation (torch.nn.functional): activation function for hidden layers
            output_activation (torch.nn.functional): activation function for output layer
        '''
        super().__init__()
        # Assign attributes and acivations
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # Initialize first linear layer
        self.input_linear = torch.nn.Linear(in_features = input_dim, out_features = hidden_dims[0])
        self.hidden_layers = torch.nn.ModuleList()
        # Adding Linear layers to the list of hidden layers
        for i in range(0, len(hidden_dims)-1):
            self.hidden_layers.append(torch.nn.Linear(in_features = hidden_dims[i], out_features = hidden_dims[i+1]))
        # Initialize last linear layer
        self.output_linear = torch.nn.Linear(in_features = hidden_dims[-1], out_features = output_dim)

    def forward(self, inputs):
        '''
        Implements forward pass of data through the network

        Args:
            inputs (torch.tensor): input data of dimension (batch_size, input_dim)
        '''
        # Make sure inputs are of type torch.Tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype = torch.float32)
        # Make sure input is of the right dimension
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        # Pass input through first linear layer
        inputs = self.hidden_activation(self.input_linear(inputs))
        # Pass input through list of hidden layers
        for hidden_layer in self.hidden_layers:
            inputs = self.hidden_activation(hidden_layer(inputs))
        # Pass data through last linear layer
        outputs = self.output_activation(self.output_linear(inputs))
        # Return outputs
        return outputs