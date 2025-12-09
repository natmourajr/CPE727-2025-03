#!/bin/bash
# Setup training environment on GCP GPU instance
# Run this script ON THE GCP INSTANCE after SSH'ing in

set -e

echo "=========================================="
echo "Setting up IARA Training Environment"
echo "=========================================="
echo ""

# Step 1: Update system
echo "Step 1: Updating system..."
sudo apt-get update -qq
sudo apt-get install -y git curl wget htop tmux

echo "✓ System updated"
echo ""

# Step 2: Verify GPU
echo "Step 2: Verifying GPU..."
nvidia-smi || echo "Warning: GPU not detected yet, may need a few minutes"
echo ""

# Step 3: Install uv (fast Python package manager)
echo "Step 3: Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "✓ uv installed"
echo ""

# Step 4: Clone repository
echo "Step 4: Setting up repository..."
REPO_URL="https://github.com/natmourajr/CPE727-2025-03"

cd ~
if [ ! -d "CPE727-2025-03" ]; then
    git clone $REPO_URL
    echo "✓ Repository cloned"
else
    echo "✓ Repository already exists"
fi

cd CPE727-2025-03/TrabalhoFinal/MiguelFernandesSousa/trabalho_final_CPE727

echo "✓ Repository ready"
echo ""

# Step 5: Setup Python environment
echo "Step 5: Setting up Python environment..."
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

echo "✓ Python environment ready"
echo ""

# Step 6: Verify PyTorch GPU
echo "Step 6: Verifying PyTorch GPU support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""

# Step 7: Setup data (if needed)
echo "Step 7: Data setup..."
echo "Do you need to download data? (y/n)"
read DOWNLOAD_DATA

if [ "$DOWNLOAD_DATA" = "y" ]; then
    echo "Setting up data directory..."
    mkdir -p data/downloaded_content
    echo "Please upload or download your data to: $(pwd)/data/downloaded_content/"
    echo "You can use: gcloud compute scp --recurse LOCAL_PATH INSTANCE:~/training/trabalho_final_CPE727/data/"
fi

echo ""

# Step 8: Display next steps
echo "=========================================="
echo "Setup Complete! ✓"
echo "=========================================="
echo ""
echo "Environment is ready. Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run hyperparameter tuning:"
echo "   python scripts/tune_hyperparameters.py --all-dcs"
echo ""
echo "3. Run production training:"
echo "   python scripts/train_production.py --all-dcs --max-epochs 100"
echo ""
echo "4. Monitor with tmux (recommended):"
echo "   tmux new -s training"
echo "   python scripts/train_production.py --all-dcs"
echo "   # Detach with Ctrl+B then D"
echo "   # Reattach with: tmux attach -t training"
echo ""
echo "5. View MLflow results:"
echo "   mlflow ui --host 0.0.0.0 --port 5000"
echo "   # Then access from browser: http://INSTANCE_IP:5000"
echo ""
echo "6. Download results when done:"
echo "   # On your local machine:"
echo "   gcloud compute scp --recurse iara-training-gpu:~/CPE727-2025-03/TrabalhoFinal/MiguelFernandesSousa/trabalho_final_CPE727/experiments . --zone=us-central1-a"
echo ""
echo "=========================================="
echo "Important Reminders"
echo "=========================================="
echo ""
echo "- Use tmux to keep training running if you disconnect"
echo "- Monitor GPU usage with: nvidia-smi"
echo "- Check disk space with: df -h"
echo "- Stop instance when done to avoid charges!"
echo ""
