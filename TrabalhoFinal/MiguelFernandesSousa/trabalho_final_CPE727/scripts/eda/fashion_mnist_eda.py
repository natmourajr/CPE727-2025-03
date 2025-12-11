"""
Exploratory Data Analysis for Fashion MNIST Dataset

Performs comprehensive EDA including:
- Dataset overview
- Image visualization
- Pixel analysis
- Feature correlation analysis
- PCA analysis

Outputs are saved to eda/outputs/fashion_mnist/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
from PIL import Image

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import FashionMNISTLoader
from src.config import RANDOM_SEED, FASHION_MNIST_CLASS_NAMES

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "outputs" / "fashion_mnist"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
DPI = 300


def save_statistics(stats_dict, filename="statistics_summary"):
    """Save statistics to both TXT and CSV formats"""
    # Save as TXT (human readable)
    txt_path = OUTPUT_DIR / f"{filename}.txt"
    with open(txt_path, 'w') as f:
        for key, value in stats_dict.items():
            f.write(f"{key}: {value}\n")

    # Save as CSV (structured)
    csv_path = OUTPUT_DIR / f"{filename}.csv"
    df = pd.DataFrame([stats_dict])
    df.to_csv(csv_path, index=False)

    print(f"✓ Statistics saved to {txt_path} and {csv_path}")


def plot_dataset_overview(loader):
    """1. Dataset Overview"""
    print("\n" + "="*60)
    print("1. DATASET OVERVIEW")
    print("="*60)

    info = loader.get_dataset_info()

    stats = {
        "total_samples": info["train_samples"] + info["test_samples"],
        "train_samples": info["train_samples"],
        "test_samples": info["test_samples"],
        "num_classes": info["num_classes"],
        "image_height": info["image_shape"][0],
        "image_width": info["image_shape"][1],
        "feature_dim": info["feature_dim"],
    }

    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    print(f"Image shape: {info['image_shape']}")
    print(f"Feature dimension: {info['feature_dim']}")

    # Get data for class distribution
    train_data = loader.train_dataset.data.numpy()
    train_targets = loader.train_dataset.targets.numpy()

    # Class distribution (text output only - perfectly balanced dataset)
    unique, counts = np.unique(train_targets, return_counts=True)

    print("\nClass distribution:")
    for i, class_name in enumerate(FASHION_MNIST_CLASS_NAMES):
        print(f"  {class_name}: {counts[i]} samples")
        stats[f"train_{class_name.lower().replace(' ', '_').replace('/', '_')}_count"] = int(counts[i])

    return train_data, train_targets, stats


def visualize_samples(train_data, train_targets):
    """2. Image Visualization"""
    print("\n" + "="*60)
    print("2. IMAGE VISUALIZATION")
    print("="*60)

    # a) Sample grid: 10x10 grid (100 samples)
    print("\nCreating 10×10 sample grid...")
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle("Fashion MNIST: 100 Random Samples", fontsize=16, fontweight='bold', y=0.995)

    random_indices = np.random.choice(len(train_data), 100, replace=False)
    for i, ax in enumerate(axes.flat):
        idx = random_indices[i]
        ax.imshow(train_data[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_grid.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: sample_grid.png")

    # b) Class representatives: 10 samples per class
    print("Creating class representative samples...")
    fig, axes = plt.subplots(10, 10, figsize=(16, 15))
    fig.suptitle("Fashion MNIST: 10 Samples per Class", fontsize=16, fontweight='bold', y=0.995)

    for class_idx in range(10):
        class_mask = train_targets == class_idx
        class_indices = np.where(class_mask)[0]
        sample_indices = np.random.choice(class_indices, 10, replace=False)

        for i in range(10):
            ax = axes[class_idx, i]
            ax.imshow(train_data[sample_indices[i]], cmap='gray')
            ax.axis('off')

            # Add class label to the first column
            if i == 0:
                # Add text label on the left side of the first image
                ax.text(-0.5, 0.5, FASHION_MNIST_CLASS_NAMES[class_idx],
                       transform=ax.transAxes,
                       fontsize=11, fontweight='bold',
                       ha='right', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "samples_per_class.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: samples_per_class.png")


def analyze_pixel_intensities(train_data, train_targets, stats):
    """3. Pixel Analysis - a) Intensity Distributions"""
    print("\n" + "="*60)
    print("3. PIXEL ANALYSIS")
    print("="*60)
    print("\n--- a) Intensity Distributions ---")

    # Flatten all pixels
    all_pixels = train_data.flatten()

    # Raw [0, 255] distribution
    stats["pixel_min_raw"] = int(all_pixels.min())
    stats["pixel_max_raw"] = int(all_pixels.max())
    stats["pixel_mean_raw"] = float(all_pixels.mean())
    stats["pixel_std_raw"] = float(all_pixels.std())

    print(f"Raw pixel range: [{stats['pixel_min_raw']}, {stats['pixel_max_raw']}]")
    print(f"Raw pixel mean: {stats['pixel_mean_raw']:.2f}")
    print(f"Raw pixel std: {stats['pixel_std_raw']:.2f}")

    # Normalized [-1, 1] distribution
    pixels_normalized = (all_pixels.astype(np.float32) / 255.0) * 2 - 1
    stats["pixel_min_normalized"] = float(pixels_normalized.min())
    stats["pixel_max_normalized"] = float(pixels_normalized.max())
    stats["pixel_mean_normalized"] = float(pixels_normalized.mean())
    stats["pixel_std_normalized"] = float(pixels_normalized.std())

    print(f"Normalized pixel range: [{stats['pixel_min_normalized']:.2f}, {stats['pixel_max_normalized']:.2f}]")

    # Visualization (3x2 grid to include log scale)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Raw distribution (linear scale)
    ax = axes[0, 0]
    ax.hist(all_pixels, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Pixel Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Pixel Intensity Distribution (Raw [0, 255])", fontsize=12, fontweight='bold')
    ax.axvline(stats["pixel_mean_raw"], color='red', linestyle='--', linewidth=2,
              label=f'Mean={stats["pixel_mean_raw"]:.1f}')
    ax.legend()

    # Raw distribution (log scale)
    ax = axes[0, 1]
    ax.hist(all_pixels, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Pixel Value", fontsize=11)
    ax.set_ylabel("Frequency (log scale)", fontsize=11)
    ax.set_yscale('log')
    ax.set_title("Pixel Intensity Distribution (Raw [0, 255]) - Log Scale", fontsize=12, fontweight='bold')
    ax.axvline(stats["pixel_mean_raw"], color='red', linestyle='--', linewidth=2,
              label=f'Mean={stats["pixel_mean_raw"]:.1f}')
    ax.legend()

    # Normalized distribution (linear scale)
    ax = axes[1, 0]
    ax.hist(pixels_normalized, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel("Pixel Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Pixel Intensity Distribution (Normalized [-1, 1])", fontsize=12, fontweight='bold')
    ax.axvline(stats["pixel_mean_normalized"], color='red', linestyle='--', linewidth=2,
              label=f'Mean={stats["pixel_mean_normalized"]:.2f}')
    ax.legend()

    # Normalized distribution (log scale)
    ax = axes[1, 1]
    ax.hist(pixels_normalized, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel("Pixel Value", fontsize=11)
    ax.set_ylabel("Frequency (log scale)", fontsize=11)
    ax.set_yscale('log')
    ax.set_title("Pixel Intensity Distribution (Normalized [-1, 1]) - Log Scale", fontsize=12, fontweight='bold')
    ax.axvline(stats["pixel_mean_normalized"], color='red', linestyle='--', linewidth=2,
              label=f'Mean={stats["pixel_mean_normalized"]:.2f}')
    ax.legend()

    # Per-class mean pixel intensity
    ax = axes[2, 0]
    class_means = []
    for i in range(10):
        class_mask = train_targets == i
        class_pixels = train_data[class_mask].flatten()
        class_means.append(class_pixels.mean())

    bars = ax.bar(range(10), class_means, color=sns.color_palette("husl", 10))
    ax.set_xticks(range(10))
    ax.set_xticklabels(FASHION_MNIST_CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel("Mean Pixel Intensity", fontsize=11)
    ax.set_title("Mean Pixel Intensity by Class", fontsize=12, fontweight='bold')

    # Per-class std pixel intensity
    ax = axes[2, 1]
    class_stds = []
    for i in range(10):
        class_mask = train_targets == i
        class_pixels = train_data[class_mask].flatten()
        class_stds.append(class_pixels.std())

    bars = ax.bar(range(10), class_stds, color=sns.color_palette("husl", 10))
    ax.set_xticks(range(10))
    ax.set_xticklabels(FASHION_MNIST_CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel("Pixel Intensity Std Dev", fontsize=11)
    ax.set_title("Pixel Intensity Std Dev by Class", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pixel_intensity_distribution.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pixel_intensity_distribution.png")


def analyze_spatial_patterns(train_data, train_targets):
    """3. Pixel Analysis - b) Spatial Analysis"""
    print("\n--- b) Spatial Analysis ---")

    # Pixel variance heatmap (across all images)
    print("Computing pixel variance heatmap...")
    pixel_variance = train_data.var(axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Overall variance heatmap
    ax = axes[0, 0]
    im = ax.imshow(pixel_variance, cmap='hot', aspect='auto')
    ax.set_title("Pixel Variance (All Images)", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Average image (all classes)
    ax = axes[0, 1]
    avg_image = train_data.mean(axis=0)
    im = ax.imshow(avg_image, cmap='gray', aspect='auto')
    ax.set_title("Average Image (All Classes)", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Per-class variance (example: first class)
    ax = axes[0, 2]
    class_0_mask = train_targets == 0
    class_0_variance = train_data[class_0_mask].var(axis=0)
    im = ax.imshow(class_0_variance, cmap='hot', aspect='auto')
    ax.set_title(f"Pixel Variance ({FASHION_MNIST_CLASS_NAMES[0]})", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Average images for 5 sample classes
    for i in range(5):
        ax = axes[1, i % 3 if i < 3 else i - 3]
        class_mask = train_targets == i
        class_avg = train_data[class_mask].mean(axis=0)
        im = ax.imshow(class_avg, cmap='gray', aspect='auto')
        ax.set_title(f"Avg: {FASHION_MNIST_CLASS_NAMES[i]}", fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pixel_variance_heatmap.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pixel_variance_heatmap.png")

    # Create separate plot comparing clothing vs footwear variance
    print("Computing pixel variance for clothing vs footwear...")

    # Clothing types (tops): T-shirt, Pullover, Coat, Shirt
    clothing_indices = [0, 2, 4, 6]
    # Footwear types: Sandal, Sneaker, Ankle boot
    footwear_indices = [5, 7, 9]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Pixel Variance: Clothing (Tops) vs Footwear", fontsize=14, fontweight='bold')

    # Top row: Clothing variance heatmaps
    for i, class_idx in enumerate(clothing_indices):
        ax = axes[0, i]
        class_mask = train_targets == class_idx
        class_variance = train_data[class_mask].var(axis=0)
        im = ax.imshow(class_variance, cmap='hot', aspect='auto')
        ax.set_title(f"{FASHION_MNIST_CLASS_NAMES[class_idx]}", fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: Footwear variance heatmaps (3 shoes + 1 combined)
    for i, class_idx in enumerate(footwear_indices):
        ax = axes[1, i]
        class_mask = train_targets == class_idx
        class_variance = train_data[class_mask].var(axis=0)
        im = ax.imshow(class_variance, cmap='hot', aspect='auto')
        ax.set_title(f"{FASHION_MNIST_CLASS_NAMES[class_idx]}", fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Combined footwear variance
    ax = axes[1, 3]
    footwear_mask = np.isin(train_targets, footwear_indices)
    footwear_variance = train_data[footwear_mask].var(axis=0)
    im = ax.imshow(footwear_variance, cmap='hot', aspect='auto')
    ax.set_title("All Footwear Combined", fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pixel_variance_clothing_vs_footwear.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pixel_variance_clothing_vs_footwear.png")

    # Create separate plot for all average images per class
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Average Image per Class", fontsize=14, fontweight='bold')

    for i in range(10):
        ax = axes[i // 5, i % 5]
        class_mask = train_targets == i
        class_avg = train_data[class_mask].mean(axis=0)
        im = ax.imshow(class_avg, cmap='gray', aspect='auto')
        ax.set_title(FASHION_MNIST_CLASS_NAMES[i], fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_images.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: average_images.png")


def analyze_feature_correlation(train_data, train_targets, stats):
    """4. Feature Correlation Analysis"""
    print("\n" + "="*60)
    print("4. FEATURE CORRELATION ANALYSIS")
    print("="*60)

    # Flatten images to (n_samples, 784)
    X_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)

    # Normalize to [0, 1] for correlation analysis
    X_flat = X_flat / 255.0

    # a) Local pixel correlation (neighboring pixels)
    print("\n--- Local Pixel Correlation ---")

    # Horizontal neighbors
    horizontal_corrs = []
    for i in range(28):
        for j in range(27):
            pixel1 = train_data[:, i, j].astype(np.float32)
            pixel2 = train_data[:, i, j+1].astype(np.float32)
            corr = np.corrcoef(pixel1, pixel2)[0, 1]
            if not np.isnan(corr):
                horizontal_corrs.append(corr)

    avg_horizontal_corr = np.mean(horizontal_corrs)
    stats["avg_horizontal_pixel_corr"] = avg_horizontal_corr
    print(f"Average horizontal neighbor correlation: {avg_horizontal_corr:.4f}")

    # Vertical neighbors
    vertical_corrs = []
    for i in range(27):
        for j in range(28):
            pixel1 = train_data[:, i, j].astype(np.float32)
            pixel2 = train_data[:, i+1, j].astype(np.float32)
            corr = np.corrcoef(pixel1, pixel2)[0, 1]
            if not np.isnan(corr):
                vertical_corrs.append(corr)

    avg_vertical_corr = np.mean(vertical_corrs)
    stats["avg_vertical_pixel_corr"] = avg_vertical_corr
    print(f"Average vertical neighbor correlation: {avg_vertical_corr:.4f}")

    # b) Feature correlation statistics (on sample)
    print("\n--- Feature Correlation Statistics ---")
    print("Computing correlation on random 50-pixel subset...")

    n_features_sample = 50
    random_features = np.random.choice(X_flat.shape[1], n_features_sample, replace=False)
    X_sample = X_flat[:, random_features]

    # Correlation matrix
    corr_matrix = np.corrcoef(X_sample.T)
    corr_matrix_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_no_diag, 0)

    # Statistics
    mean_abs_corr = np.abs(corr_matrix_no_diag).mean()
    stats["mean_abs_correlation"] = mean_abs_corr
    print(f"Mean absolute correlation: {mean_abs_corr:.4f}")

    # Distribution
    corr_values = corr_matrix_no_diag[np.triu_indices_from(corr_matrix_no_diag, k=1)]

    thresholds = [0.1, 0.3, 0.5]
    for thresh in thresholds:
        pct = (np.abs(corr_values) > thresh).sum() / len(corr_values) * 100
        stats[f"corr_above_{thresh}"] = pct
        print(f"Percentage of |correlation| > {thresh}: {pct:.2f}%")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Correlation matrix heatmap
    ax = axes[0, 0]
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                ax=ax, cbar_kws={'label': 'Correlation'}, square=True)
    ax.set_title(f"Pixel Correlation Matrix\n(Random {n_features_sample} pixels)",
                 fontsize=12, fontweight='bold')

    # Correlation distribution
    ax = axes[0, 1]
    ax.hist(corr_values, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(mean_abs_corr, color='red', linestyle='--', linewidth=2,
              label=f'Mean |corr|={mean_abs_corr:.3f}')
    ax.axvline(-mean_abs_corr, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Correlation Coefficient", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Pixel Correlations", fontsize=12, fontweight='bold')
    ax.legend()

    # Local correlation comparison
    ax = axes[1, 0]
    categories = ['Horizontal\nNeighbors', 'Vertical\nNeighbors']
    values = [avg_horizontal_corr, avg_vertical_corr]
    bars = ax.bar(categories, values, color=['skyblue', 'lightcoral'])
    ax.set_ylabel("Average Correlation", fontsize=11)
    ax.set_title("Local Pixel Correlations", fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    # Correlation strength breakdown
    ax = axes[1, 1]
    bins = [0, 0.1, 0.3, 0.5, 1.0]
    hist, _ = np.histogram(np.abs(corr_values), bins=bins)
    bin_labels = ['0-0.1\n(Weak)', '0.1-0.3\n(Moderate)', '0.3-0.5\n(Strong)', '0.5-1.0\n(Very Strong)']
    percentages = hist / len(corr_values) * 100
    bars = ax.bar(bin_labels, percentages, color=sns.color_palette("RdYlGn", 4))
    ax.set_ylabel("Percentage of Pairs (%)", fontsize=11)
    ax.set_title("Distribution of Correlation Strengths", fontsize=12, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_analysis.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: correlation_analysis.png")


def analyze_pca(train_data, train_targets, stats):
    """5. PCA Analysis"""
    print("\n" + "="*60)
    print("5. PCA ANALYSIS")
    print("="*60)

    # Flatten and normalize
    X_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    X_normalized = (X_flat / 255.0) * 2 - 1  # Normalize to [-1, 1]

    # Fit PCA
    print("\nFitting PCA...")
    pca = PCA(random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_normalized)

    # Cumulative variance explained
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    # Find number of components for thresholds
    for threshold in [90, 95, 99]:
        n_components = np.argmax(cumvar >= threshold) + 1
        stats[f"pca_components_for_{threshold}pct_var"] = n_components
        print(f"Components for {threshold}% variance: {n_components}")

    # Key thresholds from proposal
    for n in [2, 10, 50, 100]:
        if n < len(cumvar):
            stats[f"pca_variance_with_{n}_components"] = cumvar[n-1]
            print(f"Variance explained by {n} components: {cumvar[n-1]:.2f}%")

    # a) Cumulative variance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=2)
    ax.axhline(90, color='red', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax.axhline(99, color='green', linestyle='--', alpha=0.7, label='99%')

    # Mark key thresholds
    for n in [2, 10, 50, 100]:
        if n < len(cumvar):
            ax.axvline(n, color='purple', linestyle=':', alpha=0.5)
            ax.text(n, cumvar[n-1] + 2, f'{n}', ha='center', fontsize=9, color='purple')

    ax.set_xlabel("Number of Components", fontsize=12)
    ax.set_ylabel("Cumulative Variance Explained (%)", fontsize=12)
    ax.set_title("PCA: Cumulative Variance Explained", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_cumulative_variance.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_cumulative_variance.png")

    # b) Scree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    variance_pct = pca.explained_variance_ratio_[:50] * 100  # First 50 components
    ax.bar(range(1, len(variance_pct) + 1), variance_pct)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    ax.set_title("PCA Scree Plot (First 50 Components)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_scree_plot.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_scree_plot.png")

    # c) Principal component visualization (eigenfaces-style)
    print("\nVisualizing principal components as images...")
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("First 16 Principal Components (Eigenfaces)", fontsize=14, fontweight='bold')

    for i in range(16):
        ax = axes[i // 4, i % 4]
        component = pca.components_[i].reshape(28, 28)
        im = ax.imshow(component, cmap='RdBu_r', aspect='auto')
        ax.set_title(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_components_visualization.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_components_visualization.png")

    # d) 2D projection
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", 10)
    for i, class_name in enumerate(FASHION_MNIST_CLASS_NAMES):
        mask = train_targets == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]],
                  label=class_name, alpha=0.5, s=5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
    ax.set_title("PCA: 2D Projection (First 2 Components)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_2d_projection.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_2d_projection.png")

    # e) Interactive 3D projection with hover images
    print("\nCreating interactive 3D plot with hover images...")
    print("(Subsampling 2000 points for reasonable file size...)")

    # Subsample for hover images (to keep HTML file size manageable)
    n_hover_samples = min(2000, len(X_pca))
    hover_indices = np.random.choice(len(X_pca), n_hover_samples, replace=False)

    # Function to convert image to base64 for hover display
    def img_to_base64(img_array):
        """Convert 28x28 numpy array to base64 encoded PNG"""
        # Scale up image for better visibility (28x28 -> 84x84)
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.resize((84, 84), Image.NEAREST)

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    # Create base64 encoded images for hover
    print("Encoding images for hover display...")
    hover_images = [img_to_base64(train_data[i]) for i in hover_indices]

    # Create DataFrame with hover images
    df_pca = pd.DataFrame({
        'PC1': X_pca[hover_indices, 0],
        'PC2': X_pca[hover_indices, 1],
        'PC3': X_pca[hover_indices, 2],
        'Class': [FASHION_MNIST_CLASS_NAMES[train_targets[i]] for i in hover_indices],
        'image': hover_images
    })

    # Create plotly figure with customdata containing images
    fig = go.Figure()

    # Add trace for each class
    colors = px.colors.qualitative.Plotly
    for i, class_name in enumerate(FASHION_MNIST_CLASS_NAMES):
        mask = df_pca['Class'] == class_name
        class_data = df_pca[mask]

        # Create customdata with images
        customdata = np.column_stack((
            class_data['image'].values,
            class_data['PC1'].values,
            class_data['PC2'].values,
            class_data['PC3'].values
        ))

        fig.add_trace(go.Scatter3d(
            x=class_data['PC1'],
            y=class_data['PC2'],
            z=class_data['PC3'],
            mode='markers',
            name=class_name,
            marker=dict(
                size=3,
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            customdata=customdata,
            hovertemplate='<b>' + class_name + '</b><br>' +
                         'PC1: %{customdata[1]:.2f}<br>' +
                         'PC2: %{customdata[2]:.2f}<br>' +
                         'PC3: %{customdata[3]:.2f}<br>' +
                         '<extra></extra>',
        ))

    # Update layout
    fig.update_layout(
        title='PCA: Interactive 3D Projection (Hover to see images)',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)'
        ),
        hovermode='closest'
    )

    # Write HTML with custom JavaScript for image hover
    html_string = fig.to_html(include_plotlyjs='cdn')

    # Add custom CSS and JavaScript for image tooltip
    custom_js = """
    <style>
    #image-tooltip {
        position: fixed;
        display: none;
        background: white;
        border: 2px solid #333;
        border-radius: 5px;
        padding: 5px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
        pointer-events: none;
        z-index: 10000;
    }
    #image-tooltip img {
        display: block;
    }
    </style>
    <div id="image-tooltip"></div>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var tooltip = document.getElementById('image-tooltip');
        var plotDiv = document.querySelector('.plotly');

        plotDiv.on('plotly_hover', function(data) {
            var point = data.points[0];
            if (point.customdata && point.customdata[0]) {
                var img = '<img src="' + point.customdata[0] + '" width="84">';
                tooltip.innerHTML = img;
                tooltip.style.display = 'block';
            }
        });

        plotDiv.on('plotly_unhover', function(data) {
            tooltip.style.display = 'none';
        });

        plotDiv.addEventListener('mousemove', function(e) {
            if (tooltip.style.display === 'block') {
                tooltip.style.left = (e.pageX + 15) + 'px';
                tooltip.style.top = (e.pageY + 15) + 'px';
            }
        });
    });
    </script>
    """

    # Insert custom JS before closing body tag
    html_string = html_string.replace('</body>', custom_js + '</body>')

    # Write modified HTML
    with open(OUTPUT_DIR / "pca_3d_projection.html", 'w') as f:
        f.write(html_string)

    print("✓ Saved: pca_3d_projection.html (interactive with hover images)")

    # f) Covariance comparison (diagonal vs full)
    print("\n--- PCA-based Covariance Comparison ---")
    print("Comparing diagonal vs full covariance on 2D PCA projection...")

    fig = plt.figure(figsize=(16, 12))

    for i, class_name in enumerate(FASHION_MNIST_CLASS_NAMES):
        ax = plt.subplot(2, 5, i + 1)
        mask = train_targets == i
        X_class = X_pca[mask, :2]

        # Plot data points (sample for visualization)
        sample_size = min(500, len(X_class))
        sample_indices = np.random.choice(len(X_class), sample_size, replace=False)
        ax.scatter(X_class[sample_indices, 0], X_class[sample_indices, 1],
                  alpha=0.2, s=5, label='Data')

        # Fit diagonal covariance Gaussian
        mean = X_class.mean(axis=0)
        cov_diag = np.diag(X_class.var(axis=0))

        # Fit full covariance Gaussian
        cov_full = np.cov(X_class.T)

        # Plot ellipses
        from matplotlib.patches import Ellipse

        # Diagonal covariance ellipse
        eigenvalues_diag = np.diag(cov_diag)
        angle_diag = 0
        width_diag, height_diag = 2 * np.sqrt(eigenvalues_diag) * 2
        ellipse_diag = Ellipse(mean, width_diag, height_diag, angle=angle_diag,
                              facecolor='none', edgecolor='red', linewidth=2,
                              linestyle='--', label='Diagonal')
        ax.add_patch(ellipse_diag)

        # Full covariance ellipse
        eigenvalues_full, eigenvectors_full = np.linalg.eig(cov_full)
        angle_full = np.degrees(np.arctan2(eigenvectors_full[1, 0], eigenvectors_full[0, 0]))
        width_full, height_full = 2 * np.sqrt(eigenvalues_full) * 2
        ellipse_full = Ellipse(mean, width_full, height_full, angle=angle_full,
                              facecolor='none', edgecolor='blue', linewidth=2,
                              label='Full')
        ax.add_patch(ellipse_full)

        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.set_title(class_name, fontsize=10, fontweight='bold')

        if i == 0:
            ax.legend(fontsize=8)

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    plt.suptitle("Diagonal vs Full Covariance Comparison (2D PCA Projection)",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_covariance_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pca_covariance_comparison.png")

    return pca


def analyze_tsne(train_data, train_targets, stats):
    """6. t-SNE Analysis"""
    print("\n" + "="*60)
    print("6. t-SNE ANALYSIS")
    print("="*60)

    # Flatten and normalize
    X_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    X_normalized = (X_flat / 255.0) * 2 - 1  # Normalize to [-1, 1]

    # Subsample for t-SNE (computationally expensive)
    print("\nSubsampling 5,000 samples for t-SNE analysis...")
    n_samples = min(5000, len(X_normalized))
    sample_indices = np.random.choice(len(X_normalized), n_samples, replace=False)
    X_sample = X_normalized[sample_indices]
    labels_sample = train_targets[sample_indices]

    # a) 2D t-SNE
    print("\nFitting 2D t-SNE (this may take a few minutes)...")
    tsne_2d = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, max_iter=1000)
    X_tsne_2d = tsne_2d.fit_transform(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", 10)
    for i, class_name in enumerate(FASHION_MNIST_CLASS_NAMES):
        mask = labels_sample == i
        ax.scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1], c=[colors[i]],
                  label=class_name, alpha=0.6, s=10)
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE: 2D Projection", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsne_2d_projection.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: tsne_2d_projection.png")

    # b) 3D t-SNE with hover images
    print("\nFitting 3D t-SNE (this may take a few minutes)...")
    tsne_3d = TSNE(n_components=3, random_state=RANDOM_SEED, perplexity=30, max_iter=1000)
    X_tsne_3d = tsne_3d.fit_transform(X_sample)

    print("Creating interactive 3D t-SNE plot with hover images...")
    print("(Subsampling 2000 points for reasonable file size...)")

    # Further subsample for hover images
    n_hover_samples = min(2000, len(X_tsne_3d))
    hover_indices = np.random.choice(len(X_tsne_3d), n_hover_samples, replace=False)

    # Function to convert image to base64 for hover display
    def img_to_base64(img_array):
        """Convert 28x28 numpy array to base64 encoded PNG"""
        # Scale up image for better visibility (28x28 -> 84x84)
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.resize((84, 84), Image.NEAREST)

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    # Get original images for hover
    print("Encoding images for hover display...")
    original_indices = sample_indices[hover_indices]
    hover_images = [img_to_base64(train_data[i]) for i in original_indices]

    # Create DataFrame with hover images
    df_tsne = pd.DataFrame({
        'tSNE1': X_tsne_3d[hover_indices, 0],
        'tSNE2': X_tsne_3d[hover_indices, 1],
        'tSNE3': X_tsne_3d[hover_indices, 2],
        'Class': [FASHION_MNIST_CLASS_NAMES[labels_sample[i]] for i in hover_indices],
        'image': hover_images
    })

    # Create plotly figure with customdata containing images
    fig = go.Figure()

    # Add trace for each class
    colors = px.colors.qualitative.Plotly
    for i, class_name in enumerate(FASHION_MNIST_CLASS_NAMES):
        mask = df_tsne['Class'] == class_name
        class_data = df_tsne[mask]

        # Create customdata with images
        customdata = np.column_stack((
            class_data['image'].values,
            class_data['tSNE1'].values,
            class_data['tSNE2'].values,
            class_data['tSNE3'].values
        ))

        fig.add_trace(go.Scatter3d(
            x=class_data['tSNE1'],
            y=class_data['tSNE2'],
            z=class_data['tSNE3'],
            mode='markers',
            name=class_name,
            marker=dict(
                size=3,
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            customdata=customdata,
            hovertemplate='<b>' + class_name + '</b><br>' +
                         't-SNE 1: %{customdata[1]:.2f}<br>' +
                         't-SNE 2: %{customdata[2]:.2f}<br>' +
                         't-SNE 3: %{customdata[3]:.2f}<br>' +
                         '<extra></extra>',
        ))

    # Update layout
    fig.update_layout(
        title='t-SNE: Interactive 3D Projection (Hover to see images)',
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        ),
        hovermode='closest'
    )

    # Write HTML with custom JavaScript for image hover
    html_string = fig.to_html(include_plotlyjs='cdn')

    # Add custom CSS and JavaScript for image tooltip
    custom_js = """
    <style>
    #image-tooltip {
        position: fixed;
        display: none;
        background: white;
        border: 2px solid #333;
        border-radius: 5px;
        padding: 5px;
        box-shadow: 0 0 10px rgba(0,0,0,0.5);
        pointer-events: none;
        z-index: 10000;
    }
    #image-tooltip img {
        display: block;
    }
    </style>
    <div id="image-tooltip"></div>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var tooltip = document.getElementById('image-tooltip');
        var plotDiv = document.querySelector('.plotly');

        plotDiv.on('plotly_hover', function(data) {
            var point = data.points[0];
            if (point.customdata && point.customdata[0]) {
                var img = '<img src="' + point.customdata[0] + '" width="84">';
                tooltip.innerHTML = img;
                tooltip.style.display = 'block';
            }
        });

        plotDiv.on('plotly_unhover', function(data) {
            tooltip.style.display = 'none';
        });

        plotDiv.addEventListener('mousemove', function(e) {
            if (tooltip.style.display === 'block') {
                tooltip.style.left = (e.pageX + 15) + 'px';
                tooltip.style.top = (e.pageY + 15) + 'px';
            }
        });
    });
    </script>
    """

    # Insert custom JS before closing body tag
    html_string = html_string.replace('</body>', custom_js + '</body>')

    # Write modified HTML
    with open(OUTPUT_DIR / "tsne_3d_projection.html", 'w') as f:
        f.write(html_string)

    print("✓ Saved: tsne_3d_projection.html (interactive with hover images)")


def compute_bhattacharyya_distance(mu1, cov1, mu2, cov2):
    """Compute Bhattacharyya distance between two multivariate Gaussians"""
    # Average covariance
    cov_avg = (cov1 + cov2) / 2

    # Compute terms
    diff = mu1 - mu2

    try:
        # First term: Mahalanobis distance using average covariance
        cov_avg_inv = np.linalg.inv(cov_avg)
        mahal_term = 0.125 * diff.T @ cov_avg_inv @ diff

        # Second term: determinant ratio
        det_avg = np.linalg.det(cov_avg)
        det_prod = np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))

        if det_prod > 0 and det_avg > 0:
            det_term = 0.5 * np.log(det_avg / det_prod)
        else:
            det_term = 0

        return mahal_term + det_term
    except:
        # If singular, use simplified Euclidean distance
        return euclidean(mu1, mu2)


def analyze_class_separability(train_data, train_targets):
    """7. Pairwise Class Separability Matrix"""
    print("\n" + "="*60)
    print("7. CLASS SEPARABILITY ANALYSIS")
    print("="*60)

    # Flatten and normalize
    X_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    X_normalized = (X_flat / 255.0) * 2 - 1

    n_classes = 10
    n_components_list = [3, 10, 50]

    # Create figure with 3 subplots (one for each PCA dimension)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    all_distance_matrices = {}

    for idx, n_comp in enumerate(n_components_list):
        print(f"\n--- Separability in {n_comp}D PCA Space ---")

        # Fit PCA
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_normalized)

        # Compute mean and covariance for each class
        print(f"Computing class statistics in {n_comp}D...")
        class_means = []
        class_covs = []

        for i in range(n_classes):
            mask = train_targets == i
            X_class = X_pca[mask]
            class_means.append(X_class.mean(axis=0))
            class_covs.append(np.cov(X_class.T) + np.eye(n_comp) * 1e-6)  # Regularization

        # Compute pairwise Bhattacharyya distances
        print(f"Computing pairwise Bhattacharyya distances...")
        distance_matrix = np.zeros((n_classes, n_classes))

        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    distance_matrix[i, j] = 0
                else:
                    dist = compute_bhattacharyya_distance(
                        class_means[i], class_covs[i],
                        class_means[j], class_covs[j]
                    )
                    distance_matrix[i, j] = dist

        all_distance_matrices[n_comp] = distance_matrix

        # Visualize separability matrix
        ax = axes[idx]

        # Create heatmap
        im = ax.imshow(distance_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=20)

        # Labels
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(FASHION_MNIST_CLASS_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(FASHION_MNIST_CLASS_NAMES, fontsize=9)

        # Annotate with values
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j:
                    text = ax.text(j, i, f'{distance_matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)

        ax.set_title(f"{n_comp}D PCA Space", fontsize=12, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Bhattacharyya Distance', fraction=0.046, pad=0.04)

        # Print most confused pairs for this dimensionality
        print(f"\nMost confused pairs ({n_comp}D):")
        pairs = []
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                pairs.append((i, j, distance_matrix[i, j]))

        pairs.sort(key=lambda x: x[2])

        for i, j, dist in pairs[:5]:
            print(f"  {FASHION_MNIST_CLASS_NAMES[i]} ↔ {FASHION_MNIST_CLASS_NAMES[j]}: {dist:.2f}")

    fig.suptitle("Pairwise Class Separability Matrix (Bhattacharyya Distance)\nComparison across PCA Dimensions",
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_separability_matrix.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: class_separability_matrix.png")

    # Create summary comparison table
    print("\n" + "="*60)
    print("SEPARABILITY COMPARISON ACROSS DIMENSIONS")
    print("="*60)

    # Compute average separability for each dimension
    for n_comp in n_components_list:
        dm = all_distance_matrices[n_comp]
        # Get upper triangle (excluding diagonal)
        upper_tri = dm[np.triu_indices_from(dm, k=1)]
        avg_sep = upper_tri.mean()
        std_sep = upper_tri.std()
        print(f"\n{n_comp}D PCA:")
        print(f"  Average pairwise distance: {avg_sep:.2f} ± {std_sep:.2f}")
        print(f"  Min distance (most confused): {upper_tri.min():.2f}")
        print(f"  Max distance (most separated): {upper_tri.max():.2f}")


def analyze_confusion_simulation(train_data, train_targets):
    """8. Confusion Simulation with k-NN"""
    print("\n" + "="*60)
    print("8. CONFUSION SIMULATION (k-NN)")
    print("="*60)

    # Flatten and normalize
    X_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    X_normalized = (X_flat / 255.0) * 2 - 1

    # Subsample for computational efficiency
    print("\nSubsampling 10,000 samples for k-NN simulation...")
    n_samples = min(10000, len(X_normalized))
    sample_indices = np.random.choice(len(X_normalized), n_samples, replace=False)
    X_sample = X_normalized[sample_indices]
    y_sample = train_targets[sample_indices]

    # Use PCA for speed
    print("Reducing to 50 PCA components...")
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_sample)

    # Split into train/test for simulation
    split_idx = int(0.7 * len(X_pca))
    X_train, X_test = X_pca[:split_idx], X_pca[split_idx:]
    y_train, y_test = y_sample[:split_idx], y_sample[split_idx:]

    # Train k-NN classifier
    print(f"Training k-NN (k=5) on {len(X_train)} samples...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predict
    print(f"Predicting on {len(X_test)} test samples...")
    y_pred = knn.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=FASHION_MNIST_CLASS_NAMES,
                yticklabels=FASHION_MNIST_CLASS_NAMES)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Raw Counts)\nk-NN (k=5) on 50D PCA',
                fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Normalized percentages
    ax = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                xticklabels=FASHION_MNIST_CLASS_NAMES,
                yticklabels=FASHION_MNIST_CLASS_NAMES,
                vmin=0, vmax=1)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)\nk-NN (k=5) on 50D PCA',
                fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_simulation_knn.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: confusion_simulation_knn.png")

    # Print most confused pairs from confusion matrix
    print("\n--- Most Confused Class Pairs (from confusion matrix) ---")
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm_normalized[i, j] > 0:
                pairs.append((i, j, cm_normalized[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, conf_rate in pairs[:15]:
        print(f"  {FASHION_MNIST_CLASS_NAMES[i]} → {FASHION_MNIST_CLASS_NAMES[j]}: {conf_rate*100:.1f}%")

    # Compute accuracy
    accuracy = (y_pred == y_test).mean() * 100
    print(f"\n✓ Overall k-NN Accuracy: {accuracy:.2f}%")


def analyze_hierarchical_clustering(train_data, train_targets):
    """9. Hierarchical Clustering Dendrogram"""
    print("\n" + "="*60)
    print("9. HIERARCHICAL CLUSTERING")
    print("="*60)

    # Flatten and normalize
    X_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    X_normalized = (X_flat / 255.0) * 2 - 1

    # Use PCA for computational efficiency
    print("\nReducing to 50 PCA components...")
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_normalized)

    # Compute class centroids
    print("Computing class centroids...")
    n_classes = 10
    centroids = []

    for i in range(n_classes):
        mask = train_targets == i
        centroid = X_pca[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Perform hierarchical clustering
    print("Performing hierarchical clustering (Ward linkage)...")
    linkage_matrix = linkage(centroids, method='ward')

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(14, 8))

    dendrogram(
        linkage_matrix,
        labels=FASHION_MNIST_CLASS_NAMES,
        ax=ax,
        leaf_font_size=12,
        color_threshold=0
    )

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (Ward)', fontsize=12, fontweight='bold')
    ax.set_title('Hierarchical Clustering of Class Centroids\n(50D PCA Space, Ward Linkage)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hierarchical_clustering_dendrogram.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: hierarchical_clustering_dendrogram.png")

    # Print clustering interpretation
    print("\n--- Hierarchical Clustering Interpretation ---")
    print("Classes that merge early (low distance) are more similar.")
    print("The dendrogram shows natural groupings in the data.")


def analyze_silhouette_scores(train_data, train_targets):
    """10. Per-Cluster Silhouette Scores"""
    print("\n" + "="*60)
    print("10. SILHOUETTE SCORE ANALYSIS")
    print("="*60)

    # Flatten and normalize
    X_flat = train_data.reshape(train_data.shape[0], -1).astype(np.float32)
    X_normalized = (X_flat / 255.0) * 2 - 1

    # Subsample for computational efficiency
    print("\nSubsampling 10,000 samples...")
    n_samples = min(10000, len(X_normalized))
    sample_indices = np.random.choice(len(X_normalized), n_samples, replace=False)
    X_sample = X_normalized[sample_indices]
    y_sample = train_targets[sample_indices]

    # Use PCA
    print("Reducing to 50 PCA components...")
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_sample)

    # Compute silhouette scores
    print("Computing silhouette scores...")
    silhouette_avg = silhouette_score(X_pca, y_sample)
    sample_silhouette_values = silhouette_samples(X_pca, y_sample)

    print(f"\n✓ Average Silhouette Score: {silhouette_avg:.4f}")
    print("(Range: -1 to +1, higher is better)")
    print("  > 0.7: Strong cluster structure")
    print("  0.5-0.7: Reasonable cluster structure")
    print("  0.25-0.5: Weak cluster structure")
    print("  < 0.25: No substantial cluster structure")

    # Compute per-class silhouette scores
    print("\n--- Per-Class Silhouette Scores ---")
    class_silhouettes = []

    for i in range(10):
        mask = y_sample == i
        class_score = sample_silhouette_values[mask].mean()
        class_silhouettes.append(class_score)
        print(f"  {FASHION_MNIST_CLASS_NAMES[i]}: {class_score:.4f}")

    # Visualizations
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Bar plot of per-class scores
    ax = axes[0]
    colors = sns.color_palette("husl", 10)
    bars = ax.bar(range(10), class_silhouettes, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(silhouette_avg, color='red', linestyle='--', linewidth=2,
              label=f'Average={silhouette_avg:.3f}')
    ax.set_xticks(range(10))
    ax.set_xticklabels(FASHION_MNIST_CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Per-Class Silhouette Scores\n(50D PCA Space)',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(class_silhouettes) * 1.2])

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{class_silhouettes[i]:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Silhouette plot (violin plot)
    ax = axes[1]

    silhouette_data = []
    class_labels = []

    for i in range(10):
        mask = y_sample == i
        class_values = sample_silhouette_values[mask]
        silhouette_data.append(class_values)
        class_labels.append(FASHION_MNIST_CLASS_NAMES[i])

    parts = ax.violinplot(silhouette_data, positions=range(10), widths=0.7,
                          showmeans=True, showmedians=False)

    ax.axhline(silhouette_avg, color='red', linestyle='--', linewidth=2,
              label=f'Average={silhouette_avg:.3f}')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(10))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score Distribution per Class',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_scores.png", dpi=DPI, bbox_inches='tight')
    plt.close()
    print("✓ Saved: silhouette_scores.png")


def generate_profiling_report(train_data, train_targets):
    """11. Generate comprehensive HTML report using ydata-profiling"""
    print("\n" + "="*60)
    print("11. GENERATING COMPREHENSIVE HTML REPORT")
    print("="*60)

    try:
        from ydata_profiling import ProfileReport

        print("\nCreating DataFrame for profiling...")

        # Sample data for profiling (full dataset is too large)
        sample_size = min(5000, len(train_data))
        sample_indices = np.random.choice(len(train_data), sample_size, replace=False)

        # Create features for profiling
        df_data = {
            'class': [FASHION_MNIST_CLASS_NAMES[train_targets[i]] for i in sample_indices],
            'mean_pixel': [train_data[i].mean() for i in sample_indices],
            'std_pixel': [train_data[i].std() for i in sample_indices],
            'min_pixel': [train_data[i].min() for i in sample_indices],
            'max_pixel': [train_data[i].max() for i in sample_indices],
            'nonzero_pixels': [(train_data[i] > 0).sum() for i in sample_indices],
        }

        df = pd.DataFrame(df_data)

        print("Generating profiling report (this may take a few minutes)...")
        profile = ProfileReport(df, title="Fashion MNIST Dataset - EDA Report",
                               explorative=True, minimal=False)

        output_path = OUTPUT_DIR / "full_eda_report.html"
        profile.to_file(output_path)
        print(f"✓ Saved: full_eda_report.html")

    except ImportError:
        print("⚠ ydata-profiling not installed, skipping HTML report generation")
    except Exception as e:
        print(f"⚠ Error generating profiling report: {e}")


def main():
    """Main execution function"""
    print("="*60)
    print("FASHION MNIST DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Initialize loader
    print("\nInitializing data loader...")
    loader = FashionMNISTLoader(flatten=False, normalize=False)

    # Initialize statistics dictionary
    stats = {}

    # 1. Dataset Overview
    train_data, train_targets, overview_stats = plot_dataset_overview(loader)
    stats.update(overview_stats)

    # 2. Image Visualization
    visualize_samples(train_data, train_targets)

    # 3a. Pixel Analysis - Intensity Distributions
    analyze_pixel_intensities(train_data, train_targets, stats)

    # 3b. Spatial Analysis
    analyze_spatial_patterns(train_data, train_targets)

    # 4. Feature Correlation Analysis
    analyze_feature_correlation(train_data, train_targets, stats)

    # 5. PCA Analysis
    analyze_pca(train_data, train_targets, stats)

    # 6. t-SNE Analysis
    analyze_tsne(train_data, train_targets, stats)

    # 7. Class Separability Analysis
    analyze_class_separability(train_data, train_targets)

    # 8. Confusion Simulation
    analyze_confusion_simulation(train_data, train_targets)

    # 9. Hierarchical Clustering
    analyze_hierarchical_clustering(train_data, train_targets)

    # 10. Silhouette Score Analysis
    analyze_silhouette_scores(train_data, train_targets)

    # 11. Generate profiling report
    generate_profiling_report(train_data, train_targets)

    # Save statistics summary
    print("\n" + "="*60)
    print("SAVING STATISTICS SUMMARY")
    print("="*60)
    save_statistics(stats)

    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
