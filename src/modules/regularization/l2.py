import torch


class L2Regularizer:
    """L2 Regularization (Ridge/Weight Decay)

    Adds L2 penalty to the loss: lambda * sum(w^2)
    Encourages small weights.

    Note: PyTorch optimizers have built-in weight_decay parameter
    that implements L2 regularization. This class is provided for
    explicit control and educational purposes.
    """

    def __init__(self, lambda_l2=0.01):
        """
        Args:
            lambda_l2: L2 regularization strength (default: 0.01)
        """
        self.lambda_l2 = lambda_l2

    def __call__(self, model):
        """Compute L2 penalty for all model parameters

        Args:
            model: PyTorch model

        Returns:
            L2 penalty term to be added to loss
        """
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.lambda_l2 * l2_loss
