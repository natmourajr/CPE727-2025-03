import numpy as np
import torch

#=====================================================================================
# Scritp to train Deep Neural Networks (DNN) and Convolutional Neural Networks (CNN)
#=====================================================================================

def train_model(train_loader, model, criterion, optimizer):
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    tp = 0
    fp = 0
    fn = 0

    for X, y in train_loader:
        
        y = y.float().unsqueeze(1)

        logits = model(X)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item()*batch_size

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        running_correct += (preds == y).sum().item()
        running_total += batch_size

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    epoch_loss = running_loss/running_total
    epoch_acc = running_correct/running_total

    precision = tp/(tp + fp + 1e-8)
    recall = tp/(tp + fn + 1e-8)

    return epoch_loss, epoch_acc, precision, recall


@torch.no_grad()
def validate_model(val_loader, model, criterion):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    tp = 0
    fp = 0
    fn = 0

    for X, y in val_loader:
        y = y.float().unsqueeze(1)

        logits = model(X)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        running_loss += loss.item()*batch_size

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        running_correct += (preds == y).sum().item()
        running_total += batch_size

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    epoch_loss = running_loss/running_total
    epoch_acc = running_correct/running_total

    precision = tp/(tp + fp + 1e-8)
    recall = tp/(tp + fn + 1e-8)

    return epoch_loss, epoch_acc, precision, recall


@torch.no_grad()
def get_test_predictions(model, dataloader, threshold=0.5):
    model.eval()

    all_targets = []
    all_preds   = []

    for X, y in dataloader:
        X = X.float()
        y = y.float().view(-1, 1)

        logits = model(X)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= threshold).float()

        all_targets.append(y.numpy().ravel())
        all_preds.append(preds.numpy().ravel())

    y_true = np.concatenate(all_targets).astype(int)
    y_pred = np.concatenate(all_preds).astype(int)
    return y_true, y_pred


def fit_model(train_loader, val_loader, model, criterion, optimizer, num_epochs):

    merit_figures = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
    }

    for epoch in range(1, num_epochs + 1):

        epoch_loss, epoch_acc, precision, recall = train_model(train_loader, model, criterion, optimizer)

        merit_figures['train_loss'].append(epoch_loss)
        merit_figures['train_acc'].append(epoch_acc)
        merit_figures['train_precision'].append(precision)
        merit_figures['train_recall'].append(recall)


        epoch_loss_val, epoch_acc_val, precision_val, recall_val = validate_model(val_loader, model, criterion)

        merit_figures['val_loss'].append(epoch_loss_val)
        merit_figures['val_acc'].append(epoch_acc_val)
        merit_figures['val_precision'].append(precision_val)
        merit_figures['val_recall'].append(recall_val)

        print(
            f"Epoch  [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | "
            f"Prec_viol: {precision:.4f} | Rec_viol: {recall:.4f} || "
            f"Val Loss: {epoch_loss_val:.4f} | Acc: {epoch_acc_val:.4f} | "
            f"Prec_viol: {precision_val:.4f} | Rec_viol: {recall_val:.4f}"
        )

    return merit_figures