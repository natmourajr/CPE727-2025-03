#!/bin/bash
# Setup GCP GPU instance for IARA training
# Run this script locally to create and configure the instance

set -e

# Configuration
PROJECT_ID="tpu-rent-mestrado"
ACCOUNT="mfsousa94@gmail.com"
ZONE="us-central1-a"
INSTANCE_NAME="iara-training-gpu"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu-debian-11"
IMAGE_PROJECT="deeplearning-platform-release"

echo "=========================================="
echo "GCP GPU Instance Setup for IARA Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Account: $ACCOUNT"
echo "  Zone: $ZONE"
echo "  Instance: $INSTANCE_NAME"
echo "  GPU: $GPU_TYPE x$GPU_COUNT"
echo "  Disk: $BOOT_DISK_SIZE"
echo ""

# Step 1: Authenticate and set project
echo "Step 1: Setting up GCP authentication..."
gcloud auth login $ACCOUNT 2>/dev/null || echo "Already authenticated"
gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

echo "✓ Authentication configured"
echo ""

# Step 2: Enable required APIs
echo "Step 2: Enabling required APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable serviceusage.googleapis.com

echo "✓ APIs enabled"
echo ""

# Step 3: Check GPU quota
echo "Step 3: Checking GPU quota..."
echo "If this fails, you need to request quota increase:"
echo "https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
echo ""

# Step 4: Create firewall rule for SSH (if not exists)
echo "Step 4: Configuring firewall..."
gcloud compute firewall-rules create allow-ssh \
    --allow tcp:22 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow SSH from anywhere" \
    2>/dev/null || echo "Firewall rule already exists"

echo "✓ Firewall configured"
echo ""

# Step 5: Create the instance
echo "Step 5: Creating GPU instance..."
echo "This may take 2-3 minutes..."
echo ""

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-standard \
    --metadata=install-nvidia-driver=True \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server,https-server

echo ""
echo "✓ Instance created successfully!"
echo ""

# Step 6: Wait for instance to be ready
echo "Step 6: Waiting for instance to be ready..."
sleep 30
echo "✓ Instance should be ready"
echo ""

# Step 7: Get instance details
echo "=========================================="
echo "Instance Details"
echo "=========================================="
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "Instance Name: $INSTANCE_NAME"
echo "External IP: $EXTERNAL_IP"
echo "Zone: $ZONE"
echo ""

# Step 8: Display connection info
echo "=========================================="
echo "Connection Instructions"
echo "=========================================="
echo ""
echo "1. SSH into the instance:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "2. Or use standard SSH:"
echo "   ssh -i ~/.ssh/google_compute_engine $EXTERNAL_IP"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "After connecting, run the setup script:"
echo "   # Clone the repository"
echo "   git clone https://github.com/natmourajr/CPE727-2025-03"
echo "   cd CPE727-2025-03/TrabalhoFinal/MiguelFernandesSousa/trabalho_final_CPE727"
echo "   bash scripts/setup_training_env.sh"
echo ""
echo "=========================================="
echo "Cost Estimation"
echo "=========================================="
echo "Hourly cost (approximate):"
echo "  n1-standard-4: \$0.19/hour"
echo "  T4 GPU: \$0.35/hour"
echo "  Disk (100GB): \$0.01/hour"
echo "  Total: ~\$0.55/hour"
echo ""
echo "Estimated cost for full experiments:"
echo "  8 hours × \$0.55 = ~\$4.40"
echo ""
echo "=========================================="
echo "Important: Don't Forget to Stop!"
echo "=========================================="
echo ""
echo "To stop the instance (keeps disk, stops charges):"
echo "   gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete the instance (removes everything):"
echo "   gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Setup complete! ✓"
