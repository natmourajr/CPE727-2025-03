import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron with one hidden layer"""

    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output
        """
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
