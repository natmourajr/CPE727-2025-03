import torch

#=====================================================================================
# Scritp to train Sparse Autoencoders (SAE)
#=====================================================================================

def train_sparse_autoencoder(train_loader, model, recon_criterion, optimizer, beta= 1e-3, rho=0.05):
    model.train()

    running_loss = 0.0
    running_recon = 0.0
    running_sparse = 0.0
    running_total = 0

    for X, _ in train_loader:
        X = X.float()

        x_recon, z = model(X)
        recon_loss = recon_criterion(x_recon, X)
        sparse_loss = kl_sparsity_loss(z, rho)

        loss = recon_loss + beta * sparse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        running_loss   += loss.item()        * batch_size
        running_recon  += recon_loss.item()  * batch_size
        running_sparse += sparse_loss.item() * batch_size
        running_total  += batch_size

    epoch_loss   = running_loss   / running_total
    epoch_recon  = running_recon  / running_total
    epoch_sparse = running_sparse / running_total

    return epoch_loss, epoch_recon, epoch_sparse


@torch.no_grad()
def validate_sparse_autoencoder(val_loader, model, recon_criterion, beta= 1e-3, rho=0.05):
    model.eval()

    running_loss = 0.0
    running_recon = 0.0
    running_sparse = 0.0
    running_total = 0

    for X, _ in val_loader:
        X = X.float()

        x_recon, z = model(X)
        recon_loss = recon_criterion(x_recon, X)
        sparse_loss = kl_sparsity_loss(z, rho)

        loss = recon_loss + beta * sparse_loss

        batch_size = X.size(0)
        running_loss   += loss.item()        * batch_size
        running_recon  += recon_loss.item()  * batch_size
        running_sparse += sparse_loss.item() * batch_size
        running_total  += batch_size

    epoch_loss   = running_loss   / running_total
    epoch_recon  = running_recon  / running_total
    epoch_sparse = running_sparse / running_total

    return epoch_loss, epoch_recon, epoch_sparse


def fit_sparse_autoencoder(train_loader, val_loader, model, recon_criterion, optimizer,
                           num_epochs, beta=1e-3, rho=0.05):

    history = {
        "train_loss":   [],
        "train_recon":  [],
        "train_sparse": [],
        "val_loss":     [],
        "val_recon":    [],
        "val_sparse":   [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_recon, train_sparse = train_sparse_autoencoder(
            train_loader, model, recon_criterion, optimizer, beta, rho
        )
        val_loss, val_recon, val_sparse = validate_sparse_autoencoder(
            val_loader, model, recon_criterion, beta, rho
        )

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_sparse"].append(train_sparse)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_sparse"].append(val_sparse)

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_loss:.6f} "
            f"(Recon: {train_recon:.6f}, Sparse: {train_sparse:.6f}) | "
            f"Val Loss: {val_loss:.6f} "
            f"(Recon: {val_recon:.6f}, Sparse: {val_sparse:.6f})"
        )

    return history


def explore_latent_influence(test_loader, model, latent_size):
    model.eval()

    latent_influence = torch.zeros([latent_size])
    for X, y in test_loader:
        x_recon, z = model(X)

        abs_acts = z.abs()
        _, topk_idx = abs_acts.topk(5, dim=1)

        mask = torch.zeros_like(z)
        mask.scatter_(1, topk_idx, 1.0)

        mask_violations = mask[(y == 1)]

        latent_influence += mask_violations.sum(dim=0)

    return latent_influence


@torch.no_grad()
def analyze_top_latent_impact(test_loader, model, hist, delta=0.5):
    model.eval()

    idx_main_latent_impact = int(hist.argmax().item())

    impact_sum = None
    count = 0

    for X, y in test_loader:

        X_pos = X[(y == 1)].float()

        x_base, z = model(X_pos)

        z_mod = z.clone()
        z_mod[:, idx_main_latent_impact] += delta

        x_mod = model.decode(z_mod)

        diff = (x_mod - x_base).abs()

        if impact_sum is None:
            impact_sum = torch.zeros(diff.size(1))

        impact_sum += diff.sum(dim=0)
        count += diff.size(0)

    if count == 0:
        return idx_main_latent_impact, None, 0

    mean_recon_impact = impact_sum / count

    return idx_main_latent_impact, mean_recon_impact, count


def kl_sparsity_loss(z, rho=0.05):
    rho_h = z.mean(dim=0)
    rho_h = torch.clamp(rho_h, 1e-6, 0.99999)

    KL = rho*torch.log(rho/rho_h) + (1 - rho)*torch.log((1 - rho)/(1 - rho_h))

    return KL.sum()


def show_top_impacted_features(mean_recon_impact, feature_names, top_n=20):

    values, idx = torch.topk(mean_recon_impact, k=top_n)

    print("\n\n Main impacted reconstructions:")
    for v, i in zip(values, idx):
        print(f"({i:02d}) {feature_names[i]}: {v.item():.6f}")