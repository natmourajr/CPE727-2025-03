import torch.nn as nn


class Dropout(nn.Module):
    """Dropout regularization layer

    Randomly zeroes some elements of the input tensor with probability p
    during training using samples from a Bernoulli distribution.

    This is a simple wrapper around nn.Dropout for consistency with
    the other regularization modules.

    Example:
        # In model definition:
        self.dropout = Dropout(dropout_rate=0.5)

        # In forward pass:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc2(x)
    """

    def __init__(self, dropout_rate=0.5):
        """
        Args:
            dropout_rate: Probability of an element to be zeroed (default: 0.5)
        """
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Apply dropout to input

        Args:
            x: Input tensor

        Returns:
            Output tensor with dropout applied
        """
        return self.dropout(x)
