from training.train_autoencoder import fit_autoencoder, plot_latent_3d
from dataloaders.dataloader import build_dataloaders
from models.ae import AutoEncoder

import os
import torch

from torch import nn


train_loader, val_loader, test_loader, _ = build_dataloaders(
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'MT_dsa_history_data_with_labels.csv'),
    feature_exclude = ['StmTime', 'ThrCode', 'VltCode'],
    target_col = 'ThrCode',
    batch_size = 64,
    train_size = 0.7,
    val_size = 0.15,
    random_state = 42
)

in_features = next(iter(train_loader))[0].shape[1]

model = AutoEncoder(in_features, num_l1=128, num_l2=32, num_l3=3)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = fit_autoencoder(train_loader, val_loader, model, criterion, optimizer, num_epochs=100)

plot_latent_3d(model, test_loader)