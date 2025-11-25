# Wine MLP Experiment

Experiment training and generating comprative graphs from different CNNs in MNIST dataset

## Overview

This experiment demonstrates:
- Using the MNIST dataloader
- Training a selected CNNs (AlexNet and EfficientNet_B0)
- Create CSVs containing the training metrics
- Generate graphs using the CSVs

## Local Execution

### Install Dependencies

```bash
cd src/experiments/CNNsComparisonMnistExperiment
uv sync
```

### Run the Experiment

```bash
uv run cnns_comparison_mnist_experiment --epochs 50
```

### Available Options

```bash
uv run cnns_comparison_mnist_experiment --help
```

Options:
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)
- `--models`: define the model to be use in training (default: ["alexnet", "efficientnet_b0"])
- `--dataset_fraction`: Define a fraction of the training dataset to be use (default: None) -- If None will use full dataset
- `--device`: Device to use, 'cpu' or 'cuda' (default: None) -- If None use agnostic code to decide the device

### Example with Custom Parameters

```bash
uv run cnns_comparison_mnist_experiment --epochs 100 --batch-size 64 --learning-rate 0.01
```

## Docker Execution

### Build the Docker Image

From the experiment directory:

```bash
cd src/experiments/CNNsComparisonMnistExperiment
./build_image.sh
```

This will build an image tagged as `cnnscomparisonmnistexperiment:latest`.

You can specify a custom tag:

```bash
./build_image.sh v1.0
# or
./build_image.sh myrepo/cnnscomparison:v1.0
```

### Run with Docker (CPU)

```bash
docker run -it --rm cnnscomparisonmnistexperiment:latest
```

### Run with Docker (GPU)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
docker run -it --rm --gpus all cnnscomparisonmnistexperiment:latest cnns_comparison_mnist_experiment --device cuda
```

### Run with Custom Parameters

```bash
docker run -it --rm cnnscomparisonmnistexperiment:latest cnns_comparison_mnist_experiment --epochs 100 --batch-size 64
```

## Output

The experiment creates timestamped log files in the `tmp/data/mnist/` directory:

The directory contains:
- Images for each selected metrics comparing the selected models (accuracy, test loss, timedelta and train loss)
- One directory for each selected model containing a CSV with the training data


## Dependencies

- Python 3.12
- PyTorch 2.9+
- MNIST Loader (local package)
- matplotlib
- numpy
- pandas
