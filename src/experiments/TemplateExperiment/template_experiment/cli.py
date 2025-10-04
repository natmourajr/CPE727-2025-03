from typer import Typer
from template_loader.loader import TemplateDataset
from torch.utils.data import DataLoader

app = Typer()


@app.command()
def display_batches(
    batch_size: int = 4,
    shuffle: bool = True,
    max_batches: int = 10,
):
    """Display batches of the template dataset"""

    dataset = TemplateDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: {batch}")
        if batch_idx >= max_batches:
            break


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