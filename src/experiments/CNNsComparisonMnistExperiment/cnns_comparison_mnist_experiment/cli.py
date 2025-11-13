from typer import Typer

app = Typer()


@app.command()
def train_and_compare_cnns_mnist(
    models: list[str] = ["alexnet", "efficientnet_b0"],
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    dataset_fraction=None,
    device: str | None = None
):
    from .train import train_and_compare_cnns_mnist as train_func
    train_func(
        models=models,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset_fraction=dataset_fraction,
        device=device,
    )


if __name__ == "__main__":
    app()
