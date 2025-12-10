"""Utility functions for DAE experiments."""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


def calculate_mse(output, target):
    """Calculate Mean Squared Error."""
    return torch.nn.functional.mse_loss(output, target).item()


def calculate_ssim(output, target):
    """Calculate SSIM for batch of images."""
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    ssim_values = []
    for i in range(output_np.shape[0]):
        img_out = output_np[i].squeeze()
        img_target = target_np[i].squeeze()
        ssim_val = ssim(img_target, img_out, data_range=1.0)
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)


def calculate_accuracy(predictions, targets):
    """Calculate classification accuracy."""
    pred_labels = torch.argmax(predictions, dim=1)
    correct = (pred_labels == targets).sum().item()
    total = targets.size(0)
    return correct / total

def get_device():
    """Get available device (GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_metrics_as_artifact(metrics_dict, artifact_path):
    """Save large metrics dictionary as JSON artifact."""
    import json
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
