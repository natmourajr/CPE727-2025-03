from typer import Typer
import json

app = Typer()


@app.command()
def train_breast_cancer_mlp(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    hidden_size: int = 64,
    feature_strategy: str = 'onehot',
    target_strategy: str = 'binary',
    handle_missing: str = 'drop',
    device: str = 'cpu',
    scheduler: str = 'CosineAnnealingLR',
    scheduler_params: str = '{}',
):
    """Train SimpleMLP on Breast Cancer dataset with preprocessing.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        hidden_size: Size of hidden layer.
        feature_strategy: Feature encoding strategy ('onehot', 'label', 'ordinal', 'mixed').
        target_strategy: Target encoding strategy ('binary', 'label').
        handle_missing: Missing value strategy ('drop', 'most_frequent', 'constant').
        device: Device to train on ('cpu' or 'cuda').
        scheduler: Name of the learning rate scheduler to use.
        scheduler_params: JSON string with parameters for the scheduler.
    """
    from .train import train_breast_cancer_mlp as train_func

    # Safely parse scheduler params from JSON string
    try:
        scheduler_params_dict = json.loads(scheduler_params)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string for --scheduler-params")

    train_func(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        feature_strategy=feature_strategy,
        target_strategy=target_strategy,
        handle_missing=handle_missing,
        device=device,
        scheduler_name=scheduler,
        scheduler_params=scheduler_params_dict,
    )


if __name__ == "__main__":
    app()
