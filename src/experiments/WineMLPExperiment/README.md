# Wine MLP Experiment

Experiment training SimpleMLP on the Wine Quality dataset with L1 regularization.

## Overview

This experiment demonstrates:
- Using the Wine Quality dataloader
- Training a simple MLP model (SimpleMLP)
- Applying L1 regularization
- Logging results to `results/` directory

## Local Execution

### Install Dependencies

```bash
cd src/experiments/WineMLPExperiment
uv sync
```

### Run the Experiment

```bash
uv run wine-mlp-experiment --epochs 50
```

### Available Options

```bash
uv run wine-mlp-experiment --help
```

Options:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--hidden-size`: Hidden layer size (default: 64)
- `--lambda-l1`: L1 regularization strength (default: 0.001)
- `--device`: Device to use, 'cpu' or 'cuda' (default: 'cpu')

### Example with Custom Parameters

```bash
uv run wine-mlp-experiment --epochs 100 --batch-size 64 --lambda-l1 0.01
```

## Docker Execution

### Build the Docker Image

From the experiment directory:

```bash
cd src/experiments/WineMLPExperiment
./build_image.sh
```

This will build an image tagged as `winemlpexperiment:latest`.

You can specify a custom tag:

```bash
./build_image.sh v1.0
# or
./build_image.sh myrepo/winemlp:v1.0
```

### Run with Docker (CPU)

```bash
docker run -it --rm winemlpexperiment:latest
```

### Run with Docker (GPU)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
docker run -it --rm --gpus all winemlpexperiment:latest wine-mlp-experiment --device cuda
```

### Run with Custom Parameters

```bash
docker run -it --rm winemlpexperiment:latest wine-mlp-experiment --epochs 100 --batch-size 64
```

### Save Results from Docker

To save experiment logs to your host machine:

```bash
docker run -it --rm -v $(pwd)/results:/app/results winemlpexperiment:latest
```

This mounts the local `results/` directory into the container, so logs are saved to your host.

## Output

The experiment creates timestamped log files in the `results/` directory:

```
results/wine_mlp_l1_YYYYMMDD_HHMMSS.log
```

Each log contains:
- Experiment configuration
- Training progress (every 10 epochs)
- Final training and test losses
- Model architecture details

## Architecture

- **Model**: SimpleMLP (1 hidden layer, 833 parameters)
- **Dataset**: UCI Wine Quality (6,497 samples, 80/20 train/test split)
- **Loss**: MSE Loss + L1 Regularization
- **Optimizer**: Adam
- **Features**: 11 physicochemical properties
- **Target**: Wine quality score (3-9)

## Dependencies

- Python 3.13
- PyTorch 2.6+
- Wine Quality Loader (local package)
- scikit-learn
- numpy
- pandas
