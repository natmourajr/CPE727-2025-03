import sys
import os
import time
from pathlib import Path
import json
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Agg")

# Add paths for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src/models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src/modules'))

import torch
from torch import nn, optim
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F
from train_funcs import train_loop, evaluate_test, save_model, load_checkpoint, find_best_threshold_youden

# Import covidx_cxr4 loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src/dataloaders/CovidxCxr4Loader'))
from covidx_cxr4_loader.loader import build_loaders

from DenseNet.densenet import DenseNet121
from ViT.vit import ViTBinaryClassifier
from Unet.unet import UNetBinaryClassifier

# Import logger
from logger import ExperimentLogger
from early_stop.early_stop import EarlyStopping

MODELS_NAME = {
    'densenet121': DenseNet121,
    'vit': ViTBinaryClassifier, 
    'unet': UNetBinaryClassifier
}

MODELS_TRAIN_TRANSFORMS = {
    'densenet121': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.02,0.02)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]),
    'vit': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.02,0.02)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ]),
    'unet': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.02,0.02)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2)
        ])
}

MODELS_TEST_TRANSFORMS = {
    "densenet121": transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]),
    "vit": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]),
    'unet': transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])
}


def train_and_compare_cnns_covidx_cxr4(
    models = ['unet', 'densenet121'],
    epochs=200,
    batch_size=16,
    learning_rate=0.00001,
    device=None,
):

    # Initialize logger
    repo_root = os.path.join(os.path.dirname(__file__), '../../..')
    results_dir = os.path.abspath(os.path.join(repo_root, 'results'))
    logger = ExperimentLogger('cnn_covidx_cxr4', results_dir=results_dir)

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f'Using device: {device}')

    for model_name in models:
        if model_name not in MODELS_NAME:
            logger.log(f'Unknown model: {model_name}. Skipping...')
            continue

       # --------------------------------------------------
        # Model
        # --------------------------------------------------
        model = MODELS_NAME[model_name](num_classes=1, device=device)

        # --------------------------------------------------
        # Transforms
        # --------------------------------------------------
        train_transform = MODELS_TRAIN_TRANSFORMS[model_name]
        test_transform = MODELS_TEST_TRANSFORMS[model_name]

        # --------------------------------------------------
        # Loaders (train / val / test)
        # --------------------------------------------------
        train_loader, val_loader, test_loader, pos_weight = build_loaders(
            batch_size=batch_size,
            num_workers=4,
            train_transforms=train_transform,
            test_transforms=test_transform
        )

        logger.log(f'Train batches: {len(train_loader)}')
        logger.log(f'Val batches: {len(val_loader)}')
        logger.log(f'Test batches: {len(test_loader)}')
        logger.log(f'Pos Weight: {pos_weight}')

        # --------------------------------------------------
        # Loss / Optimizer / Scheduler
        # --------------------------------------------------
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=15,
        )

        # --------------------------------------------------
        # Paths / Logger
        # --------------------------------------------------
        path = os.path.join(repo_root, 'tmp', 'covidx_cxr4', model_name)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=50,
            min_delta=0.0,
            min_mode=False,
            path=path
        )

        model, start_epoch = load_checkpoint(
            target_dir=os.path.join(path, 'checkpoints'),
            model=model,
            device=device
        )

        if start_epoch == 0:
            with open(os.path.join(path, 'training.csv'), "w") as f:
                f.write(
                    "epoch,train_loss,val_loss,roc_auc,epoch_time\n"
                )

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------
        train_loop(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            early_stopping=early_stopping,
            start_epoch=start_epoch,
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            path=path,
            logger=logger,
        )

        early_stopping.restore(model)
        threshold, score = find_best_threshold_youden(model, val_loader, device)
        logger.log(f"Best Threshold: {threshold}")
        logger.log(f"Threshold Score: {score}")
        with open(os.path.join(path, "threshold.txt"), "w") as f:
            f.write(f"threshold;score\n")
            f.write(f"{threshold};{score}\n")

        # --------------------------------------------------
        # FINAL TEST (ONCE)
        # --------------------------------------------------
        # test_metrics, _, _ = evaluate_test(
        #     model=model,
        #     test_loader=test_loader,
        #     device=device,
        #     threshold=threshold,
        #     logger=logger,
        #     path_dir=path,
        #     max_saliency_images=100
        # )
        # with open(os.path.join(path, 'test_evaluation.json'), "w") as f:
        #     json.dump(test_metrics, f)

        # Save final model
        # save_model(
        #     model=model,
        #     target_dir=os.path.join(path, 'checkpoints'),
        #     model_name="final_model.pth",
        # )

    logger.close()


if __name__ == '__main__':
    train_and_compare_cnns_covidx_cxr4()