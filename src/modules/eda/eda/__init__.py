"""EDA module for PyTorch datasets

Dataset-agnostic exploratory data analysis.
"""
from .adapters import dataset_to_dataframe, get_dataset_info
from .detector import detect_data_type, is_tabular, is_image, is_sequence
from .profile_report import generate_profile_report, generate_comparison_report
from typing import Optional, List
from torch.utils.data import Dataset


def analyze_dataset(
    train_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    output: str = "report.html",
    dataset_name: str = "Dataset",
    feature_names: Optional[List[str]] = None,
    target_name: str = "target",
    minimal: bool = False,
    explorative: bool = False,
):
    """Analyze any PyTorch dataset and generate EDA report

    Args:
        train_dataset: PyTorch Dataset (training split)
        test_dataset: PyTorch Dataset (test split, optional)
        output: Output file path for HTML report
        dataset_name: Name for report title
        feature_names: List of feature names (optional, auto-generated if None)
        target_name: Name for target column (default: 'target')
        minimal: If True, generates minimal report (faster)
        explorative: If True, generates detailed explorative report (slower)
    """
    print(f"Analyzing dataset: {dataset_name}")

    # Get dataset info
    info = get_dataset_info(train_dataset)
    print(f"  Train samples: {info['n_samples']}")

    if test_dataset:
        test_info = get_dataset_info(test_dataset)
        print(f"  Test samples: {test_info['n_samples']}")

    # Detect data type
    data_type = detect_data_type(train_dataset)
    print(f"  Detected data type: {data_type}")

    if data_type == "tabular":
        print("  Converting to pandas DataFrame...")

        # Convert to pandas
        train_df = dataset_to_dataframe(
            train_dataset,
            feature_names=feature_names,
            target_name=target_name
        )

        if test_dataset:
            test_df = dataset_to_dataframe(
                test_dataset,
                feature_names=feature_names,
                target_name=target_name
            )

            # Combine for single report
            train_df['split'] = 'train'
            test_df['split'] = 'test'
            combined_df = pd.concat([train_df, test_df], ignore_index=True)

            print("  Generating profile report (train + test)...")
            report = generate_profile_report(
                combined_df,
                title=f"{dataset_name} EDA Report",
                minimal=minimal,
                explorative=explorative
            )
        else:
            print("  Generating profile report...")
            report = generate_profile_report(
                train_df,
                title=f"{dataset_name} EDA Report",
                minimal=minimal,
                explorative=explorative
            )

        # Save report
        print(f"  Saving report to {output}...")
        report.to_file(output)
        print(f"✓ Report saved to: {output}")

    elif data_type == "image":
        print("✗ Image dataset EDA not yet implemented")

    elif data_type == "sequence":
        print("✗ Sequence dataset EDA not yet implemented")

    else:
        print(f"✗ Unsupported data type: {data_type}")


import pandas as pd

__all__ = [
    "analyze_dataset",
    "dataset_to_dataframe",
    "get_dataset_info",
    "detect_data_type",
    "is_tabular",
    "is_image",
    "is_sequence",
    "generate_profile_report",
    "generate_comparison_report",
]
