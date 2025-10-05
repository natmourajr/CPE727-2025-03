"""Generate EDA report for Wine Quality dataset

This script demonstrates how to use the EDA module programmatically.
Run this from the EDA module directory after installing wine-quality-loader.
"""
import sys
from pathlib import Path

# Add dataloaders to path
repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
wine_loader_path = repo_root / "src" / "dataloaders" / "WineQualityLoader"
sys.path.insert(0, str(wine_loader_path))

# Import after adding to path
from wine_quality_loader import WineQualityDataset
from eda import analyze_dataset

if __name__ == "__main__":
    # Load datasets
    print("Loading Wine Quality datasets...")
    train_dataset = WineQualityDataset(split='train', train_ratio=0.8, random_state=42)
    test_dataset = WineQualityDataset(split='test', train_ratio=0.8, random_state=42)

    # Feature names
    feature_names = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ]

    # Output path
    output_path = repo_root / "results" / "eda" / "wine_quality_report.html"

    # Generate report
    analyze_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output=str(output_path),
        dataset_name="Wine Quality",
        feature_names=feature_names,
        target_name="quality"
    )
