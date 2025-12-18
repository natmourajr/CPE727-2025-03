import torch
import os
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)


def load_checkpoint(target_dir, model, device):
    checkpoints_paths = sorted(list(Path(target_dir).rglob('checkpoint*')))
    if len(checkpoints_paths) == 0:
        return model, 0

    last_checkpoint = None
    i = 0
    for checkpoint in checkpoints_paths:
        checkpoint_i = int(checkpoint.name.split('_')[-1].split('.')[0]) + 1
        if checkpoint_i > i:
            i = checkpoint_i
            last_checkpoint = checkpoint

    model.load_state_dict(torch.load(last_checkpoint, map_location=torch.device(device)))

    return model, i


def train_loop(model, optimizer, criterion, scheduler, early_stopping, start_epoch, epochs, train_loader, val_loader, device, path, logger):
    # Training loop
    logger.log('=== Training ===')
    for epoch in tqdm(range(start_epoch, epochs)):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_loader)):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if len(y_batch.shape) == 1:
                y_batch = y_batch.unsqueeze(1).float()

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Average loss
        avg_train_loss = train_loss / len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in tqdm(enumerate(val_loader)):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                if len(y_batch.shape) == 1:
                    y_batch = y_batch.unsqueeze(1).float()

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                # Probabilidades contínuas (para ROC)
                probs = torch.sigmoid(outputs)

                all_probs.append(probs.view(-1).cpu())
                all_labels.append(y_batch.view(-1).cpu())


        epoch_time = time.time() - start_time
        avg_test_loss = test_loss / len(val_loader)

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # ROC
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_test_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        # Avisar caso tenha mudado
        if new_lr != old_lr:
            print(f"[Scheduler] Learning rate reduzido: {old_lr:.6f} → {new_lr:.6f}")

        # Log metrics
        logger.log_metrics(
            epoch=f'{epoch+1}/{epochs}',
            metrics_dict={
                'Train Loss': avg_train_loss,
                'Test Loss': avg_test_loss,
                'ROC-AUC': roc_auc,
                'Epoch Time (s)': epoch_time
            }
        )

        with open(os.path.join(path, 'training.csv'), 'a+') as f:
            f.write(f'{epoch+1},{avg_train_loss},{avg_test_loss},{roc_auc},{epoch_time}\n')

        if epoch % 1 == 0:
            save_model(
                model=model,
                target_dir=os.path.join(path, 'checkpoints'),
                model_name=f'checkpoint_epoch_{epoch}.pth'
            )

        early_stopping(roc_auc, model)
        if early_stopping.should_stop:
            print("Early stopping triggered.")
            break


    epoch_time = time.time() - start_time
    logger.log('')
    logger.log('=== Training Complete ===')
    logger.log(f'Final Train Loss: {avg_train_loss:.4f}')
    logger.log(f'Final Test Loss: {avg_test_loss:.4f}')
    logger.log(f'Final Test ROC-AUC: {roc_auc:.2f}%')


def find_best_threshold_youden(model, loader, device):
    labels, probs = collect_probs_labels(model, loader, device)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]

def collect_probs_labels(model, test_loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(test_loader)):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().view(-1)

            logits = model(X_batch).squeeze(1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(y_batch.cpu())

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy()
    )


def evaluate_test(model, test_loader, device, threshold, logger, path_dir=None, max_saliency_images=5):
    model.eval()
    all_probs, all_labels = [], []
    saliency_count = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device).float().view(-1)

            logits = model(X).squeeze(1)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

            # ---------- SALIENCY ----------
            if path_dir is not None and saliency_count < max_saliency_images:
                # Precisamos de grad → sair do no_grad
                with torch.enable_grad():
                    sal = generate_saliency(
                        model,
                        X[:1].clone(),   # uma imagem por vez
                        device
                    )

                os.makedirs(path_dir, exist_ok=True)

                plt.imshow(sal[0], cmap="hot")
                plt.axis("off")
                plt.title("Saliency Map")
                plt.savefig(
                    os.path.join(path_dir, f"saliency_{saliency_count}.png"),
                    bbox_inches="tight"
                )
                plt.close()

                img = X[0].detach().cpu()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                img = img.clamp(0, 1)
                img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(
                    os.path.join(path_dir, f"original_{saliency_count}.png")
                )

                saliency_count += 1

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    plot_roc(labels, probs, path_dir)

    preds = (probs >= threshold).astype(int)

    test_metrics = {
        "ROC-AUC": roc_auc_score(labels, probs),
        "Accuracy": accuracy_score(labels, preds),
        "Recall": recall_score(labels, preds),
        "Precision": precision_score(labels, preds),
        "F1": f1_score(labels, preds),
    }

    logger.log(f"Test Evaluation:\n{test_metrics}")

    return test_metrics, all_probs, all_labels

def plot_roc(labels, probs, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    roc_path = os.path.join(
        path if path else ".",
        "roc_curve.png"
    )

    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()


def generate_saliency(model, x, device):
    model.eval()

    x = x.to(device)
    x.requires_grad_()

    logits = model(x).squeeze(1)
    score = logits.sum()

    model.zero_grad()
    score.backward()

    saliency = x.grad.abs()                  # [B, C, H, W]
    saliency, _ = saliency.max(dim=1)        # [B, H, W]
    return saliency.detach().cpu()