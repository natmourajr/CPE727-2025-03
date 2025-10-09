from typer import Typer

app = Typer()


@app.command()
def train_breast_cancer_mlp(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    feature_strategy: str = 'onehot',
    target_strategy: str = 'binary',
    handle_missing: str = 'drop',
    device: str = 'cpu',
):
    """Train SimpleMLP on Breast Cancer dataset with preprocessing

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_size: Size of hidden layer
        feature_strategy: Feature encoding strategy ('onehot', 'label', 'ordinal', 'mixed')
        target_strategy: Target encoding strategy ('binary', 'label')
        handle_missing: Missing value strategy ('drop', 'most_frequent', 'constant')
        device: Device to train on ('cpu' or 'cuda')
    """
    from .train import train_breast_cancer_mlp as train_func
    train_func(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        feature_strategy=feature_strategy,
        target_strategy=target_strategy,
        handle_missing=handle_missing,
        device=device,
    )


if __name__ == "__main__":
    app()
