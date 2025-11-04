# CIFAR-10 Training - Learning Curves

Train MLP, CNN, and ResNet-18 models on CIFAR-10 and generate learning curves.

## Quick Start (Recommended)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed

### Run with Docker

```bash
# Build and run
docker compose up

# Or manually:
docker build -t cifar10-training .
docker run --rm \
  -v "$(pwd)/results":/app/results \
  -v "$(pwd)/data":/app/data \
  cifar10-training
```

### View Progress

```bash
# Watch logs in real-time
docker compose logs -f

# Check running containers
docker ps

# Stop training
docker compose down
```

### Results

Training results will be saved to `./results/<timestamp>/`:
- `*_training_curves.png` - Individual model learning curves
- `model_comparison.png` - Comparison of all models
- `*.pth` - Model checkpoints
- `summary.json` - Training metrics

## Configuration

Edit `main.py` to customize:
- Learning rate: `learning_rate = 0.001`
- Batch size: `batch_size = 64`
- Epochs: `max_epochs = 50`

## Expected Training Time

- **First run**: ~5 minutes (Docker build)
- **Training on CPU**: ~60-80 minutes
- **Training on GPU**: ~5-10 minutes (requires GPU passthrough)

## Troubleshooting

**macOS Sequoia 15 users**: If you get PyTorch symbol errors when trying to run locally, use Docker. See `documentation/MACOS_SEQUOIA_FIX.md` for details.

**Slow training**: Expected on CPU. Docker provides a consistent environment regardless of your system.

## Project Structure

```
.
├── README.md           # This file
├── docker-compose.yml  # Docker Compose config
├── Dockerfile          # Docker image definition
├── main.py             # Main training script
├── requirements.txt    # Python dependencies
├── data/               # Dataset loaders
├── models/             # Model architectures (MLP, CNN, ResNet)
├── training/           # Training logic
├── visualization/      # Plotting utilities
├── results/            # Training outputs (generated)
└── documentation/      # Detailed documentation
    ├── README.md                # Full documentation
    ├── SETUP_INSTRUCTIONS.md    # Alternative setup methods
    └── MACOS_SEQUOIA_FIX.md     # macOS-specific fixes
```

## What It Does

1. Downloads CIFAR-10 dataset (~170MB, auto-cached)
2. Trains three models:
   - **MLP**: Hidden layers [512, 256]
   - **CNN**: 64 filters
   - **ResNet-18**: Standard architecture
3. Generates training curves and comparison plots
4. Saves model checkpoints

## Device Support

The code automatically uses the best available device:
- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon)
- **CPU** (fallback)

In Docker, CPU is used by default. For GPU support, see `documentation/README.md`.

## Documentation

- **[Full Documentation](documentation/README.md)** - Complete project details
- **[Setup Instructions](documentation/SETUP_INSTRUCTIONS.md)** - Alternative installation methods
- **[macOS Sequoia Fix](documentation/MACOS_SEQUOIA_FIX.md)** - Troubleshooting for macOS 15

## Next Steps

- Phase 1.1: Implement Early Stopping
- Phase 1.2: Double Descent experiments
- Phase 2: Advanced regularization techniques

---

**TL;DR:** Run `docker compose up` and results appear in `./results/`
