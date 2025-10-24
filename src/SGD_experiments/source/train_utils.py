import torch
import torch.nn as nn
from tqdm import tqdm 

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Performs one full pass over the training data with a progress bar."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # ----------------------------------------------------
    # WRAP THE DATALOADER WITH TQDM
    # ----------------------------------------------------
    pbar = tqdm(dataloader, desc='Train Loss', unit='batch', leave=False)
    
    for inputs, labels in pbar: # <-- Use the wrapped pbar
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update the progress bar description with the current batch loss
        pbar.set_description(f"Train Loss: {loss.item():.4f}") 
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    """Performs one full evaluation pass over validation or test data."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # ----------------------------------------------------
        # WRAP THE DATALOADER WITH TQDM
        # ----------------------------------------------------
        pbar = tqdm(dataloader, desc='Eval Loss', unit='batch', leave=False)
        
        for inputs, labels in pbar: # <-- Use the wrapped pbar
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_description(f"Eval Loss: {loss.item():.4f}")

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = correct / total
    return val_loss, val_acc