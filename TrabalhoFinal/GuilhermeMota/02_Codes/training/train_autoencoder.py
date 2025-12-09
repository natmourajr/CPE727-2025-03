import matplotlib.pyplot as plt
import numpy as np
import torch

#=====================================================================================
# Scritp to train Autoencoders (AE)
#=====================================================================================

def train_autoencoder(train_loader, model, criterion, optimizer):
    model.train()

    running_loss = 0.0
    running_total = 0

    for X, _ in train_loader:
        X = X.float()

        x_recon, z = model(X)
        loss = criterion(x_recon, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        running_loss += loss.item()*batch_size
        running_total += batch_size

    epoch_loss = running_loss/running_total

    return epoch_loss


@torch.no_grad()
def validate_autoencoder(val_loader, model, criterion):
    model.eval()

    running_loss = 0.0
    running_total = 0

    for X, _ in val_loader:
        X = X.float()

        x_recon, z = model(X)
        loss = criterion(x_recon, X)

        batch_size = X.size(0)
        running_loss += loss.item()*batch_size
        running_total += batch_size

    epoch_loss = running_loss/running_total

    return epoch_loss


def fit_autoencoder(train_loader, val_loader, model, criterion, optimizer, num_epochs):

    merit_figures = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(1, num_epochs + 1):
        train_loss = train_autoencoder(train_loader, model, criterion, optimizer)
        val_loss = validate_autoencoder(val_loader, model, criterion)

        merit_figures['train_loss'].append(train_loss)
        merit_figures['val_loss'].append(val_loss)

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

    return merit_figures


def train_ae_dnn(train_loader, ae, dnn, criterion, optimizer):
    ae.eval()
    dnn.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    tp = 0
    fp = 0
    fn = 0

    for X, y in train_loader:
        X = X.float()
        y = y.float().unsqueeze(1)

        with torch.no_grad():
            z = ae.encode(X)

        logits = dnn(z)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == y).sum().item()
            running_total += batch_size

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    precision = tp/(tp + fp + 1e-8)
    recall = tp/(tp + fn + 1e-8)

    return epoch_loss, epoch_acc, precision, recall


@torch.no_grad()
def validate_ae_dnn(val_loader, ae, dnn, criterion):
    ae.eval()
    dnn.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    tp = 0
    fp = 0
    fn = 0

    for X, y in val_loader:
        X = X.float()
        y = y.float().unsqueeze(1)

        z = ae.encode(X)
        logits = dnn(z)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == y).sum().item()
        running_total += batch_size

        tp += ((preds == 1) & (y == 1)).sum().item()
        fp += ((preds == 1) & (y == 0)).sum().item()
        fn += ((preds == 0) & (y == 1)).sum().item()

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    precision = tp/(tp + fp + 1e-8)
    recall = tp/(tp + fn + 1e-8)

    return epoch_loss, epoch_acc, precision, recall


def fit_ae_dnn_classifier(train_loader, val_loader, ae, dnn, criterion, optimizer, num_epochs=20):

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
        train_loss, train_acc, train_precision, train_recall = train_ae_dnn(train_loader, ae, dnn, criterion, optimizer)
        val_loss, val_acc, val_precision, val_recall = validate_ae_dnn(val_loader, ae, dnn, criterion)

        merit_figures['train_loss'].append(train_loss)
        merit_figures['train_acc'].append(train_acc)
        merit_figures['train_precision'].append(train_precision)
        merit_figures['train_recall'].append(train_recall)

        merit_figures['val_loss'].append(val_loss)
        merit_figures['val_acc'].append(val_acc)
        merit_figures['val_precision'].append(val_precision)
        merit_figures['val_recall'].append(val_recall)

        print(
            f"Epoch  [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
            f"Prec_viol: {train_precision:.4f} | Rec_viol: {train_recall:.4f} || "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
            f"Prec_viol: {val_precision:.4f} | Rec_viol: {val_recall:.4f}"
        )

    return merit_figures


@torch.no_grad()
def get_test_predictions(ae, dnn, dataloader, threshold=0.5):
    ae.eval()
    dnn.eval()

    all_targets = []
    all_preds   = []

    for X, y in dataloader:
        X = X.float()
        y = y.float().view(-1, 1)

        x_recon, z = ae(X)
        logits = dnn(z)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= threshold).float()

        all_targets.append(y.numpy().ravel())
        all_preds.append(preds.numpy().ravel())

    y_true = np.concatenate(all_targets).astype(int)
    y_pred = np.concatenate(all_preds).astype(int)
    return y_true, y_pred


@torch.no_grad()
def plot_latent_3d(model, test_loader, title="Espaço Latente (z1, z2 e z3)"):
    model.eval()
    device = next(model.parameters()).device

    zs = []
    ys = []

    for X, y in test_loader:
        X = X.float().to(device)
        y = y.cpu().numpy()

        _, z = model(X)
        zs.append(z.cpu().numpy())
        ys.append(y)

    Z = np.concatenate(zs, axis=0)
    Y = np.concatenate(ys, axis=0)

    mask0 = (Y == 0)
    mask1 = (Y == 1)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(Z[mask0, 0], Z[mask0, 1], Z[mask0, 2],
               alpha=0.5, label="Normal", s=20)
    ax.scatter(Z[mask1, 0], Z[mask1, 1], Z[mask1, 2],
               alpha=0.7, label="Violado", s=20, marker="x")

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return Z, Y