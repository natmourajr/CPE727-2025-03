# Wine Quality Loader

DataLoader for the UCI Wine Quality dataset.

## Dataset

The [Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) from UCI ML Repository contains physicochemical and sensory data for red and white wine variants of the Portuguese "Vinho Verde" wine.

- **Total Samples**: 6,497 wine samples
- **Train Samples**: 5,197 (80% split)
- **Test Samples**: 1,300 (20% split)
- **Features**: 11 physicochemical properties
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target**: Quality score (range: 3-9 in practice, though scale is 0-10)

## Usage

```python
from wine_quality_loader.loader import WineQualityDataset
from torch.utils.data import DataLoader

# Create train and test datasets
train_dataset = WineQualityDataset(split="train", train_ratio=0.8, random_state=42)
test_dataset = WineQualityDataset(split="test", train_ratio=0.8, random_state=42)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Iterate over batches
for X_batch, y_batch in train_loader:
    # X_batch: (batch_size, 11) - features
    # y_batch: (batch_size, 1) - quality scores
    pass
```

## Parameters

- `split`: Either `"train"` or `"test"` (default: `"train"`)
- `train_ratio`: Proportion of data for training (default: `0.8`)
- `random_state`: Random seed for reproducible splits (default: `42`)

## Data Format

- **Input (X)**: FloatTensor of shape `(11,)` containing the 11 physicochemical features
- **Output (y)**: FloatTensor of shape `(1,)` containing the quality score
- **Data types**: Both X and y are `torch.float32`

## Example Output

```
Train samples: 5197
Test samples: 1300
Feature shape: torch.Size([11])
Target shape: torch.Size([1])
Quality range: [3.0, 9.0]
```

## Running Tests

Using uv:
```bash
uv run --extra dev pytest
```

Using pip:
```bash
pip install -e '.[dev]'
pytest
```

## Using in Docker

This dataloader is designed to be used as a local dependency in Docker-based experiments.

**Working Example:** See `src/experiments/WineMLPExperiment/` for a complete implementation.

### Setup in your experiment

**1. Declare dependency in `pyproject.toml`:**

```toml
# src/experiments/YourExperiment/pyproject.toml
[project]
dependencies = [
    "wine-quality-loader",
    # ... other dependencies
]

[tool.uv.sources]
wine-quality-loader = { path = "../../dataloaders/WineQualityLoader", editable = true }
```

**2. Configure Dockerfile:**

```dockerfile
# src/experiments/YourExperiment/Dockerfile
ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

FROM $BASE_IMAGE

# Set timezone
ENV TZ=Etc/UTC

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /uvx /bin/

# Copy entire repository (needed for local dependencies)
COPY . /app

WORKDIR /app/src/experiments/YourExperiment

# Setup UV and Python environment
RUN uv venv /venv
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV UV_PYTHON=/venv/bin/python
ENV PATH=/venv/bin:$PATH

# Install dependencies (including local packages)
RUN uv sync --locked

# Default command: run the experiment
CMD ["your-experiment-command", "--arg", "value"]
```

**Key points:**
- `ARG BASE_IMAGE` allows building with different base images (CPU/GPU)
- `COPY . /app` copies the entire repository (required for local dataloader path)
- `uv sync --locked` installs wine-quality-loader from the relative path
- Build Docker image from repository root using `build_image.sh`

**Reference implementation:**
- Dockerfile: `src/experiments/WineMLPExperiment/Dockerfile:11-23`
- pyproject.toml: `src/experiments/WineMLPExperiment/pyproject.toml:7-8,23-24`

## Notes

- The dataset is automatically downloaded from UCI ML Repository on first use
- `ucimlrepo` handles caching internally, so subsequent runs don't re-download
- The same `random_state` ensures consistent train/test splits across runs
- Both red and white wine variants are included in the dataset
