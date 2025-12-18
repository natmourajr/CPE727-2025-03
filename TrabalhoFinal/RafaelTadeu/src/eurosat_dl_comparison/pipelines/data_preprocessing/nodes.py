"""Data preprocessing nodes."""
import torch
import logging
import pandas as pd

from torchvision import transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, random_split

from sklearn.model_selection import StratifiedKFold
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def download_eurosat(params: Dict[str, Any]) -> EuroSAT:
    """Download and load the EuroSAT dataset.

    Args:
        params: Dataset parameters containing root path and download flag.

    Returns:
        EuroSAT dataset object.
    """
    logger.info("Downloading EuroSAT dataset...")

    # Basic transform for initial loading
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = EuroSAT(
        root=params["root"],
        download=params["download"],
        transform=basic_transform,
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples, {len(dataset.classes)} classes")
    logger.info(f"Classes: {dataset.classes}")

    return dataset


def split_dataset(
    dataset: EuroSAT,
    params: Dict[str, Any]
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Split dataset into train, validation, and test sets.

    Args:
        dataset: Full EuroSAT dataset.
        params: Split parameters (dev_ratio, seed).

    Returns:
        Tuple of (dev_dataset, test_dataset).
    """
    logger.info("Splitting dataset...")

    dev_ratio = params["dev_ratio"]
    seed = params["seed"]

    total_size = len(dataset)
    dev_size = int(dev_ratio * total_size)
    test_size = total_size - dev_size

    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    dev_dataset, test_dataset = random_split(
        dataset,
        [dev_size, test_size],
        generator=generator
    )

    logger.info(f"Split sizes - dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

    return dev_dataset, test_dataset


def create_cross_validation_dataset(
    dev_dataset: torch.utils.data.Dataset,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """Create stratified cross-validation dataset from development set.

    Uses StratifiedKFold to ensure each fold maintains the same proportional
    class distribution as the original dataset.

    Args:
        dev_dataset: Development dataset to split into folds.
        params: Cross-validation parameters (n_splits, shuffle, seed).

    Returns:
        DataFrame with columns: sample_idx, image_path, label, class_name, fold.
    """


    logger.info("Creating stratified cross-validation dataset...")

    n_splits = params["n_splits"]
    shuffle = params["shuffle"]
    seed = params["seed"]

    # Extract data from development dataset
    data_records = []
    labels_list = []

    for idx in range(len(dev_dataset)):
        # Get the original dataset index
        original_idx = dev_dataset.indices[idx]
        _, label = dev_dataset.dataset[original_idx]

        # Get image path from the dataset
        # EuroSAT stores images in root/class_name/image.jpg
        image_path = dev_dataset.dataset.samples[original_idx][0]

        data_records.append({
            "sample_idx": idx,
            "original_idx": original_idx,
            "image_path": image_path,
            "label": label,
            "class_name": dev_dataset.dataset.classes[label]
        })
        labels_list.append(label)

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Use StratifiedKFold for proportional class distribution in each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    # Initialize fold column
    df["fold"] = -1

    # Assign folds using StratifiedKFold
    for fold_idx, (_, fold_indices) in enumerate(skf.split(df.index, labels_list)):
        df.loc[fold_indices, "fold"] = fold_idx

    logger.info(
        f"Stratified cross-validation dataset created: {len(df)} samples, "
        f"{n_splits} folds"
    )

    # Log fold distribution with class breakdown
    for fold in range(n_splits):
        fold_df = df[df["fold"] == fold]
        fold_size = len(fold_df)
        class_dist = fold_df["class_name"].value_counts().to_dict()
        logger.info(
            f"Fold {fold}: {fold_size} samples - "
            f"Class distribution: {class_dist}"
        )

    # Log overall class proportions for verification
    total_class_dist = df["class_name"].value_counts(normalize=True).to_dict()
    logger.info(f"Overall class proportions: {total_class_dist}")

    return df


def create_cv_fold_loaders(
    dev_dataset: torch.utils.data.Dataset,
    cross_val_table: pd.DataFrame,
    params: Dict[str, Any],
    fold_num: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train, validation, and test loaders for a specific cross-validation fold.

    Args:
        dev_dataset: Development dataset.
        cross_val_table: DataFrame with fold assignments.
        params: DataLoader parameters (batch_size, num_workers, etc.).
        fold_num: Which fold to use as validation (others become training).

    Returns:
        Tuple of (train_loader, val_loader) for the specified fold.
    """
    from torch.utils.data import Subset

    logger.info(f"Creating data loaders for fold {fold_num}...")

    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Get indices for this fold
    train_indices = cross_val_table[cross_val_table["fold"] != fold_num]["sample_idx"].tolist()
    val_indices = cross_val_table[cross_val_table["fold"] == fold_num]["sample_idx"].tolist()

    logger.info(f"Fold {fold_num}: {len(train_indices)} training samples, {len(val_indices)} validation samples")

    # Create subsets for train and validation
    train_subset = Subset(dev_dataset, train_indices)
    val_subset = Subset(dev_dataset, val_indices)

    # Apply transforms
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = eval_transform

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=params["batch_size"],
        shuffle=True,  # Shuffle training data
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )

    logger.info(
        f"Fold {fold_num} loaders created: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)} samples"
    )

    return train_loader, val_loader


def create_test_loader(
    test_dataset: torch.utils.data.Dataset,
    params: Dict[str, Any],
) -> DataLoader:
    """Create train, validation, and test loaders for a specific cross-validation fold.

    Args:
        test_dataset: Test dataset for final evaluation.
        params: DataLoader parameters (batch_size, num_workers, etc.).

    Returns:
        test_loader.
    """
    from torch.utils.data import Subset


    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset.dataset.transform = eval_transform

    # Create data loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )

    logger.info(
        f"Loader created:  test={len(test_loader.dataset)} samples"
    )

    return test_loader


def compute_dataset_statistics(dataset: torch.utils.data.Dataset) -> Dict[str, Any]:
    """Compute statistics about the dataset.

    Args:
        dataset: Dataset to analyze.

    Returns:
        Dictionary with dataset statistics.
    """
    logger.info("Computing dataset statistics...")

    # Get class distribution
    class_counts = {}
    for _, label in dataset:
        label_name = dataset.dataset.classes[label]
        class_counts[label_name] = class_counts.get(label_name, 0) + 1

    statistics = {
        "total_samples": len(dataset),
        "num_classes": len(dataset.dataset.classes),
        "class_names": dataset.dataset.classes,
        "class_distribution": class_counts,
    }

    logger.info(f"Dataset statistics: {statistics}")

    return statistics
