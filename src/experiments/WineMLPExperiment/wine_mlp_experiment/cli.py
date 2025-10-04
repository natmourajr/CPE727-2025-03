from typer import Typer

app = Typer()


@app.command()
def train_wine_mlp(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    lambda_l1: float = 0.001,
    device: str = 'cpu',
):
    """Train SimpleMLP on Wine Quality dataset with L1 regularization"""
    from .train import train_wine_mlp as train_func
    train_func(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lambda_l1=lambda_l1,
        device=device,
    )


if __name__ == "__main__":
    app()
