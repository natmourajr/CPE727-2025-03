# Docker Setup Instructions

This directory contains a complete regularization pipeline implementation with Docker support for easy deployment and reproducibility.

## Quick Start

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and run the container
./run_docker.sh
```

### Option 2: Manual Docker Commands
```bash
# Build the image
docker build -t regularization-pipeline .

# Run the container
docker run -p 8888:8888 -v $(pwd):/app regularization-pipeline
```

## Accessing the Notebook

Once the container is running:
1. Open your browser and go to `http://localhost:8888`
2. Use the token `regularization` to access the notebook
3. Open `regularization_pipeline.ipynb` to run the baseline overfitting demonstration

## What's Included

### Python Modules
- **models/cnn.py**: Fashion MNIST CNN implementation
- **data/fashion_mnist_loader.py**: Data loading and preprocessing
- **training/trainer.py**: Training pipeline with early stopping support

### Jupyter Notebook
- **regularization_pipeline.ipynb**: Complete baseline overfitting demonstration
  - Data loading with reduced dataset (1/20 of original)
  - CNN model training without regularization
  - Comprehensive visualization of overfitting
  - Performance analysis and summary

### Docker Configuration
- **Dockerfile**: Multi-stage build with PyTorch and Jupyter
- **docker-compose.yml**: Easy orchestration
- **run_docker.sh**: Automated build and run script
- **requirements.txt**: All necessary Python dependencies

## Expected Results

The baseline experiment should demonstrate:
- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~60-70%
- **Overfitting Gap**: ~30-35%
- **Clear visualization** of training vs validation curves diverging

## Next Steps

This baseline establishes the foundation for comparing regularization techniques:
1. L1 Regularization
2. L2 Regularization (Weight Decay)
3. Elastic Net Regularization
4. Dropout
5. Batch Normalization
6. Early Stopping
7. Data Augmentation (Albumentations)

## Troubleshooting

### Port Already in Use
```bash
# Stop any existing containers
docker-compose down

# Or use a different port
docker run -p 8889:8888 -v $(pwd):/app regularization-pipeline
```

### Permission Issues
```bash
# Make sure the script is executable
chmod +x run_docker.sh
```

### Build Issues
```bash
# Clean build
docker-compose down
docker system prune -f
docker-compose build --no-cache
```

## File Structure
```
full_pipeline/
├── README.md (this file)
├── regularization_pipeline.ipynb (main notebook)
├── Dockerfile
├── docker-compose.yml
├── run_docker.sh
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── cnn.py
├── data/
│   ├── __init__.py
│   └── fashion_mnist_loader.py
└── training/
    ├── __init__.py
    └── trainer.py
```
