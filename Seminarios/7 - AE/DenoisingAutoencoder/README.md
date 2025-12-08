# Denoising Autoencoder Experiment

Comprehensive experiment for training and evaluating Convolutional Denoising Autoencoders (CDAE) on MNIST dataset with grid search and cross-validation.

## Overview

This experiment trains a CDAE to reconstruct clean MNIST images from noisy inputs. It includes:

- **Grid Search**: Automated hyperparameter optimization
- **Cross-Validation**: 5-fold CV for robust model selection
- **MLflow Tracking**: Complete experiment tracking with nested runs
- **Visualization**: Training curves, denoising comparisons, and architecture diagrams
- **Baseline Comparison**: Gaussian filter baseline for performance benchmarking

## Experiment Pipeline

1. **Data Preparation**
   - Load MNIST dataset
   - Add Gaussian noise at specified levels
   - Create 5-fold cross-validation splits

2. **Grid Search Training**
   - Train models for each hyperparameter combination
   - Each configuration trains 5 models (one per fold)
   - Early stopping with validation loss monitoring

3. **Model Selection**
   - Identify best configuration based on average validation loss
   - Retrain best model on full training set (no validation split)

4. **Evaluation & Visualization**
   - Test set evaluation (MSE, SSIM)
   - Denoising visualization grids
   - Training curve plots
   - Comparison with Gaussian filter baseline

**Metrics:** MSE, SSIM, Visual Quality

## Installation

### Local Installation

```bash
cd "Seminarios/7 - AE/DenoisingAutoencoder"
uv sync
```

### Docker Installation

Build the Docker image:

```bash
cd "Seminarios/7 - AE/DenoisingAutoencoder"
./build_image.sh
```

This will build an image tagged as `denoisingautoencoder:latest` with all dependencies and the local `mnist-loader` package installed.

## Usage

### Running with Docker

#### Run experiment info:
```bash
docker run denoisingautoencoder:latest
```

#### Run denoising experiment:
```bash
docker run -v /path/to/Data:/app/Data \
  denoisingautoencoder:latest \
  denoising-autoencoder denoise \
  --params /app/Data/autoencoder/params/denoising_params.yaml \
  --cv-splits /app/Data/autoencoder/splits/noisy_mnist_cv_splits_noise0.3.csv
```

#### Generate cross-validation splits:
```bash
docker run -v /path/to/Data:/app/Data \
  denoisingautoencoder:latest \
  denoising-autoencoder generate-cross-validation-splits \
  --output-dir /app/Data/autoencoder/splits \
  --n-splits 5 \
  --noise-level 0.3
```

**Notes:**
- Mount the `Data` directory to persist MNIST downloads and experiment results
- The container has GPU support via NVIDIA CUDA runtime
- MLflow tracking data will be stored in the container unless you mount `mlruns/`

### Running Locally

### Run Denoising Experiment with Grid Search

```bash
denoising-autoencoder denoise --params ../../../Data/autoencoder/params/denoising_params.yaml
```

This will:
- Run grid search over all parameter combinations
- Perform 5-fold cross-validation for each configuration
- Log all results to MLflow with nested runs (parent = config, children = folds)
- Save models, training metrics, and visualizations as artifacts

### Get Experiment Information

```bash
denoising-autoencoder info
```

## Parameter Files

Parameter file is stored in `Data/autoencoder/params/denoising_params.yaml`

### Parameter Structure

```yaml
base:
  # Base parameters (used if not in grid search)
  noise_seed: 42
  seed: 42
  batch_size: 64
  latent_dim: 128
  learning_rate: 0.001
  epochs: 50
  early_stopping_patience: 10

grid:
  # Grid search parameters (each combination will be tested)
  latent_dim: [64, 128, 256]
  learning_rate: [0.001, 0.005]
  noise_level: [0.3, 0.5, 0.8]
  # Total configurations: 3 × 2 × 3 = 18 configs
  # With 5 folds each: 90 total training runs
```

### Key Parameters

- **latent_dim**: Size of the bottleneck layer (feature dimension)
- **learning_rate**: Adam optimizer learning rate
- **noise_level**: Standard deviation of Gaussian noise (0-1 scale)
- **batch_size**: Training batch size
- **epochs**: Maximum training epochs per fold
- **early_stopping_patience**: Stop if validation loss doesn't improve for N epochs

## MLflow Tracking

The experiment uses **nested MLflow runs** for organized tracking:

### Run Structure
- **Parent Runs**: One per grid search configuration (e.g., `grid_search_config_1`)
  - Logs: hyperparameter combination
  - No metrics (aggregated from children)

- **Child Runs**: One per fold (e.g., `fold_0`, `fold_1`, ..., `fold_4`)
  - Logs: fold-specific test metrics (MSE, SSIM)
  - Artifacts:
    - `model_fold_X.pth`: Trained model weights
    - `fold_X_training_metrics.json`: Training/validation losses per epoch
    - `fold_X_results.png`: Denoising visualization grid (10 samples)

### Logged Information

**Per Configuration (Parent Run):**
## Features

✅ **Grid Search**: Automated hyperparameter optimization
✅ **5-Fold Cross-Validation**: Robust model evaluation
✅ **Nested MLflow Runs**: Parent (config) + Children (folds) structure
✅ **Early Stopping**: Based on validation loss
✅ **Test Set Isolation**: Only evaluated after model selection
✅ **Comprehensive Metrics**: MSE, SSIM for image quality assessment
✅ **Efficient Logging**: Training curves stored as JSON artifacts
✅ **Visualization**: Per-fold denoising grids with 10 samples
✅ **YAML Configuration**: Flexible parameter management
✅ **GPU Acceleration**: Automatic CUDA detection
✅ **Analysis Notebooks**: Post-experiment analysis and visualization
✅ **Baseline Comparison**: Gaussian filter for benchmarking
✅ **Model Persistence**: Save best models for deployment
```bash
# Start MLflow UI
cd "Seminarios/7 - AE/DenoisingAutoencoder"
mlflow ui

# Navigate to http://localhost:5000
# Parent runs show configurations
# Click to expand and see individual fold results
```

## GPU Support

Experiments automatically use GPU if available.

## Features

✅ Cross-validation with train/val/test splits
✅ Test set isolation (only used for final scoring)
✅ MSE, SSIM, and Accuracy metrics
✅ Efficient MLflow logging (large data as artifacts)
## Project Structure

```text
Seminarios/7 - AE/DenoisingAutoencoder/
├── denoising_autoencoder/      # Main package
│   ├── __init__.py
│   ├── cli.py                  # Command-line interface
│   ├── utils.py                # Shared utilities
│   ├── denoise/
│   │   ├── __init__.py
│   │   └── experiment.py       # Main denoising experiment
│   └── models/
│       ├── __init__.py
│       ├── denoising_autoencoder.py  # Model architecture
│       └── README.md           # Model documentation
├── .notebooks/                 # Analysis notebooks
├── mlruns/                     # MLflow tracking directory
├── Dockerfile                  # Docker container configuration
├── build_image.sh              # Docker build script
├── pyproject.toml              # Project dependencies (uv)
├── uv.lock                     # Locked dependencies
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Models Used

- **ConvolutionalDenoisingAutoencoder**: Convolutional denoising autoencoder model
  - 3-block encoder with Conv2d → BatchNorm → ReLU → MaxPool
  - Latent space with configurable dimensions
  - 3-block decoder with ConvTranspose2d → BatchNorm → ReLU
  - Final Conv2d output layer with Sigmoid activation

## Data Loaders

- **MNIST Loader**: Uses `mnist-loader` package from local dataloaders
- **Noisy MNIST Generator**: Generates noisy MNIST images
  - Adds Gaussian noise: `noisy = clean + N(0, σ²)`
  - Clips values to [0, 1] range
  - Configurable noise levels (σ)

## Cross-Validation Splits

- **Train/Val Split**: 80/20 for each fold
- **Test Set**: Held out completely (never used during training/validation)
- **Split Files**: `Data/autoencoder/splits/noisy_mnist_cv_splits_noise{level}.csv`
- **Reproducibility**: Fixed seed ensures consistent splits across runs

## Analysis & Visualization

After running experiments, use the Jupyter notebook for:

1. **Training Curve Analysis**: Plot loss curves for all folds
2. **Best Model Selection**: Compare configurations and find optimal hyperparameters
3. **Retraining**: Train final model on full training set (no validation split)
4. **Baseline Comparison**: Compare against Gaussian filter
5. **Quantitative Metrics**: Calculate MSE, SSIM on test set
6. **Visual Comparison**: Side-by-side denoising results
7. **LaTeX Tables**: Generate publication-ready result tables

### Example Analysis Workflow

```python
# In Jupyter notebook
# 1. Load MLflow runs and analyze configurations
# 2. Find best config based on avg validation loss
# 3. Retrain on full training set
# 4. Compare with Gaussian filter baseline
# 5. Generate visualizations and metrics tables
```

## Output Files

After running the experiment:

- `mlruns/`: MLflow tracking data
- `model_fold_X.pth`: Trained model checkpoints (per fold)
- `fold_X_training_metrics.json`: Training history (per fold)
- `fold_X_results.png`: Denoising visualizations (per fold)
- `best_model_final.pth`: Final model trained on full dataset (from notebook)
- `grid_search_val_loss.tex`: LaTeX table with results (from notebook)

## Data Loaders

- **MNIST**: `src/dataloaders/MNISTLoader/`
- **Noisy MNIST**: `src/dataloaders/MNISTLoader/mnist_loader/noise_mnist/`
