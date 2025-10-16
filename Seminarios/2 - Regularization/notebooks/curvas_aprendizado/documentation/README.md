# CIFAR-10 Training - Learning Curves

This project trains MLP, CNN, and ResNet-18 models on CIFAR-10 and generates learning curves.

## ⚠️ macOS Sequoia 15 Issue

**If you're on macOS Sequoia 15 (26.0), PyTorch precompiled wheels are incompatible with your system.**

See `MACOS_SEQUOIA_FIX.md` for details.

## 🐳 Recommended: Docker Solution (Works Everywhere)

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))

### Quick Start with Docker

```bash
# Build and run
docker-compose up

# Or manually:
docker build -t cifar10-training .
docker run -v $(pwd)/results:/app/results -v $(pwd)/data:/app/data cifar10-training
```

Results will be saved to `./results/` directory.

## 🖥️ Local Installation (If Not on Sequoia)

### Prerequisites
- Python 3.9, 3.10, or 3.11
- NOT Python 3.12 or 3.13

### Setup

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm

# Run training
python main.py
```

## 📊 What It Does

1. **Loads CIFAR-10 Dataset** (auto-downloads if needed)
2. **Trains 3 Models:**
   - MLP (hidden: 512, 256)
   - SimpleCNN (filters: 64)
   - ResNet-18

3. **Generates Plots:**
   - Individual training curves (loss & accuracy)
   - Model comparison plots
   - Saves checkpoints

4. **Outputs to `results/<timestamp>/`:**
   - `*_training_curves.png` - Individual model curves
   - `model_comparison.png` - Compare all models
   - `*.pth` - Model checkpoints
   - `summary.json` - Training metrics

## 🎯 Device Support

The code automatically detects and uses:
- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon M1/M2/M3)
- **CPU** (fallback)

## 📁 Project Structure

```
.
├── main.py              # Main training script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker containerization
├── docker-compose.yml  # Easy Docker deployment
├── data/               # Dataset loaders
├── models/             # Model architectures
├── training/           # Training logic
├── visualization/      # Plotting utilities
└── results/            # Training outputs (created at runtime)
```

## 🔧 Configuration

Edit `main.py` to customize:
- Learning rate: `learning_rate = 0.001`
- Batch size: `batch_size = 64`
- Epochs: `max_epochs = 50`
- Model architectures

## 📝 Next Steps (Planned)

- Phase 1.1: Early Stopping
- Phase 1.2: Double Descent experiments
- Phase 2: Advanced regularization techniques

## 🐛 Troubleshooting

### "Symbol not found" errors
→ Use Docker solution (see above)

### "CUDA not available" on Mac
→ This is expected. Mac uses MPS, not CUDA. The code handles this.

### Slow training
→ Expected on CPU. Use Docker with GPU support or cloud platforms (Colab, AWS, etc.)

## 📚 References

- [PyTorch](https://pytorch.org/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

**Quick Docker Command:**
```bash
docker-compose up
```

Results appear in `./results/`

