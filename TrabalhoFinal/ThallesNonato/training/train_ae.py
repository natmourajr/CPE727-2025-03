import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ThallesNonato.models.ae import AutoRec

def train_autorec(R, dataloader, hidden_dim=500, lr=0.001, num_epochs=10, device=None):

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoRec(num_items=R.shape[1], hidden_dim=hidden_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset_size = len(dataloader.dataset)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)  
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / dataset_size
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model
