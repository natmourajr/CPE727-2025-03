# EDA Module

Flexible, dataset-agnostic exploratory data analysis for PyTorch datasets.

## Features

- **Dataset Agnostic**: Works with any PyTorch Dataset (no dataset-specific imports)
- **Automatic Detection**: Auto-detects data type (tabular, image, sequence)
- **Comprehensive Reports**: Generates HTML reports with ydata-profiling
- **CLI Support**: Easy command-line interface with dynamic dataset loading
- **Programmatic API**: Use directly in Python code or notebooks

## Installation

```bash
cd src/modules/eda
uv sync
```

## Usage

### Option 1: CLI (Recommended)

Generate EDA report from command line using dynamic dataset loading:

```bash
uv run eda-report \
  --train-dataset "wine_quality_loader:WineQualityDataset:split=train,train_ratio=0.8,random_state=42" \
  --test-dataset "wine_quality_loader:WineQualityDataset:split=test,train_ratio=0.8,random_state=42" \
  --output results/eda/wine_report.html \
  --name "Wine Quality"
```

**Dataset Spec Format:**
```
"module:ClassName:arg1=val1,arg2=val2"
```

**CLI Options:**
- `--train-dataset`: Train dataset specification (required)
- `--test-dataset`: Test dataset specification (optional)
- `--output`, `-o`: Output HTML file path (default: results/eda/report.html)
- `--name`, `-n`: Dataset name for report title (default: Dataset)
- `--features`: Comma-separated feature names (optional, auto-generated if not provided)
- `--target`: Target column name (default: target)
- `--minimal`: Generate minimal report (faster)
- `--explorative`: Generate detailed explorative report (slower)

**Examples:**

```bash
# Minimal report (faster)
uv run eda-report \
  --train-dataset "wine_quality_loader:WineQualityDataset:split=train" \
  --output results/eda/wine_minimal.html \
  --minimal

# Explorative report with feature names
uv run eda-report \
  --train-dataset "wine_quality_loader:WineQualityDataset:split=train" \
  --features "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol" \
  --target "quality" \
  --explorative
```

### Option 2: Programmatic API

Use directly in Python code or Jupyter notebooks:

```python
from eda import analyze_dataset
from wine_quality_loader import WineQualityDataset

# Load datasets
train_dataset = WineQualityDataset(split='train', train_ratio=0.8, random_state=42)
test_dataset = WineQualityDataset(split='test', train_ratio=0.8, random_state=42)

# Generate report
analyze_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    output="results/eda/wine_report.html",
    dataset_name="Wine Quality",
    feature_names=[
        "fixed_acidity", "volatile_acidity", "citric_acid",
        "residual_sugar", "chlorides", "free_sulfur_dioxide",
        "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
    ],
    target_name="quality"
)
```

## What's in the Report?

The generated HTML report includes:

- **Dataset Overview**: Number of samples, features, missing values
- **Variable Statistics**: Mean, std, min, max, quartiles for each feature
- **Distribution Plots**: Histograms for all features
- **Correlation Matrix**: Feature correlations and heatmap
- **Missing Values**: Detection and visualization
- **Duplicate Rows**: Identification of duplicates
- **Train/Test Split**: Combined analysis if both splits provided

## Supported Data Types

Currently supported:
- ✅ **Tabular**: 1D tensors or flat feature vectors (regression, classification)

Planned:
- 🔄 **Image**: 3D tensors (C, H, W)
- 🔄 **Sequence**: 2D tensors (time series, text)

## Architecture

```
src/modules/eda/
├── pyproject.toml          # Package configuration
├── README.md               # This file
└── eda/
    ├── __init__.py         # Main API (analyze_dataset)
    ├── adapters.py         # PyTorch Dataset → pandas conversion
    ├── detector.py         # Auto-detect data type
    ├── profile_report.py   # ydata-profiling wrapper
    └── cli.py              # CLI with dynamic import
```

## How It Works

1. **Load Dataset**: Dynamically imports and instantiates dataset from spec
2. **Detect Type**: Auto-detects if data is tabular, image, or sequence
3. **Convert**: Converts PyTorch Dataset to pandas DataFrame (for tabular)
4. **Profile**: Generates comprehensive report with ydata-profiling
5. **Save**: Exports interactive HTML report

## Dependencies

- Python >=3.11
- torch >=2.6.0
- pandas >=2.0.0
- ydata-profiling >=4.6.0
- typer >=0.16.0
- matplotlib >=3.8.0
- seaborn >=0.13.0

## Notes

- The module is completely dataset-agnostic - it doesn't import any specific datasets
- Works with any PyTorch Dataset that implements `__len__` and `__getitem__`
- For large datasets, use `--minimal` flag for faster report generation
- Reports are self-contained HTML files that can be opened in any browser
