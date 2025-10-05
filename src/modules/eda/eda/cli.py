"""CLI for EDA module

Provides command-line interface for generating EDA reports from any PyTorch dataset.
"""
import sys
import os
from pathlib import Path
from typer import Typer, Option
import importlib
from typing import Optional


app = Typer()


def load_dataset_from_spec(spec: str):
    """Load dataset from specification string

    Format: "module.path:ClassName:arg1=value1,arg2=value2"

    Example:
        "wine_quality_loader:WineQualityDataset:split=train,train_ratio=0.8,random_state=42"

    Args:
        spec: Dataset specification string

    Returns:
        Instantiated PyTorch Dataset
    """
    parts = spec.split(':')

    if len(parts) < 2:
        raise ValueError(
            f"Invalid dataset spec: {spec}\n"
            "Format: 'module:ClassName' or 'module:ClassName:arg1=val1,arg2=val2'"
        )

    module_path = parts[0]
    class_name = parts[1]
    kwargs = {}

    # Parse keyword arguments
    if len(parts) > 2:
        for pair in parts[2].split(','):
            if '=' not in pair:
                continue
            key, value = pair.split('=', 1)

            # Type conversion
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.replace('.', '').replace('-', '').isdigit():
                value = float(value) if '.' in value else int(value)

            kwargs[key] = value

    # Add local dataloaders to path
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    dataloaders_root = repo_root / "src" / "dataloaders"

    # Find and add specific dataloader directory
    for loader_dir in dataloaders_root.iterdir():
        if loader_dir.is_dir() and not loader_dir.name.startswith('.'):
            sys.path.insert(0, str(loader_dir))

    # Dynamic import
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}': {e}\n"
            f"Make sure the module is installed or in the correct path."
        )

    try:
        dataset_class = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(
            f"Module '{module_path}' has no class '{class_name}'"
        )

    # Instantiate dataset
    try:
        dataset = dataset_class(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating {class_name} with args {kwargs}: {e}"
        )

    return dataset


@app.command()
def eda_report(
    train_dataset_spec: str = Option(
        ...,
        "--train-dataset",
        help="Train dataset spec: 'module:Class:arg1=val1,arg2=val2'"
    ),
    test_dataset_spec: Optional[str] = Option(
        None,
        "--test-dataset",
        help="Test dataset spec (optional)"
    ),
    output: str = Option(
        "results/eda/report.html",
        "--output",
        "-o",
        help="Output file path for HTML report"
    ),
    dataset_name: str = Option(
        "Dataset",
        "--name",
        "-n",
        help="Dataset name for report title"
    ),
    feature_names: Optional[str] = Option(
        None,
        "--features",
        help="Comma-separated feature names (optional)"
    ),
    target_name: str = Option(
        "target",
        "--target",
        help="Target column name"
    ),
    minimal: bool = Option(
        False,
        "--minimal",
        help="Generate minimal report (faster)"
    ),
    explorative: bool = Option(
        False,
        "--explorative",
        help="Generate detailed explorative report (slower)"
    ),
):
    """Generate EDA report for any PyTorch dataset

    Example:
        \b
        eda-report \\
          --train-dataset "wine_quality_loader:WineQualityDataset:split=train" \\
          --test-dataset "wine_quality_loader:WineQualityDataset:split=test" \\
          --output results/eda/wine_report.html \\
          --name "Wine Quality"
    """
    from . import analyze_dataset

    # Create output directory
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading train dataset from: {train_dataset_spec}")
    train_dataset = load_dataset_from_spec(train_dataset_spec)

    test_dataset = None
    if test_dataset_spec:
        print(f"Loading test dataset from: {test_dataset_spec}")
        test_dataset = load_dataset_from_spec(test_dataset_spec)

    # Parse feature names
    features = None
    if feature_names:
        features = [f.strip() for f in feature_names.split(',')]

    # Generate report
    analyze_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output=output,
        dataset_name=dataset_name,
        feature_names=features,
        target_name=target_name,
        minimal=minimal,
        explorative=explorative,
    )


if __name__ == "__main__":
    app()
