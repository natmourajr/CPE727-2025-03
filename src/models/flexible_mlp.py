import torch.nn as nn


class FlexibleMLP(nn.Module):
    """Multi-Layer Perceptron with configurable number of hidden layers"""

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes, e.g., [64, 32, 16]
            output_size: Size of output
        """
        super(FlexibleMLP, self).__init__()

        layers = []
        prev_size = input_size

        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
