from training.train_sparse_autoencoder import fit_sparse_autoencoder, explore_latent_influence, analyze_top_latent_impact, show_top_impacted_features
from dataloaders.dataloader import build_dataloaders
from models.ae import AutoEncoder

import os
import torch

from torch import nn


train_loader, val_loader, test_loader, X_col = build_dataloaders(
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'MT_dsa_history_data_with_labels.csv'),
    feature_exclude = ['StmTime', 'ThrCode', 'VltCode'],
    target_col = 'VltCode',
    batch_size = 64,
    train_size = 0.7,
    val_size = 0.15,
    random_state = 42
)

model = AutoEncoder(next(iter(train_loader))[0].shape[1], num_l1=512, num_l2=1024, num_l3=2048, ae_type='sparse')

recon_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = fit_sparse_autoencoder(train_loader, val_loader, model, recon_criterion, optimizer, num_epochs=100, beta=1e-2, rho=0.05)

latent_influence = explore_latent_influence(test_loader, model, latent_size=2048)

idx_main_latent_impact, mean_recon_impact, count = analyze_top_latent_impact(test_loader, model, latent_influence, delta=1.0)

show_top_impacted_features(mean_recon_impact, X_col, top_n=10)