import torch


class L1Regularizer:
    """L1 Regularization (Lasso)

    Adds L1 penalty to the loss: lambda * sum(|w|)
    Encourages sparsity in model weights.
    """

    def __init__(self, lambda_l1=0.01):
        """
        Args:
            lambda_l1: L1 regularization strength (default: 0.01)
        """
        self.lambda_l1 = lambda_l1

    def __call__(self, model):
        """Compute L1 penalty for all model parameters

        Args:
            model: PyTorch model

        Returns:
            L1 penalty term to be added to loss
        """
        l1_loss = 0.0
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.lambda_l1 * l1_loss
