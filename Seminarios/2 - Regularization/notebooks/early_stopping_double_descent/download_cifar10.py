#!/usr/bin/env python3
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def download_cifar10():
    """Download CIFAR-10 dataset and save as npz format"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("Downloading CIFAR-10 dataset...")
    
    # Download training set
    trainset = CIFAR10(root='./datasets', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    
    # Download test set
    testset = CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    
    print("Processing training data...")
    # Get all training data
    for data in trainloader:
        X_train_tensor, y_train = data
        break
    
    print("Processing test data...")
    # Get all test data
    for data in testloader:
        X_test_tensor, y_test = data
        break
    
    # Convert from tensor to numpy and transpose to match expected format
    # PyTorch format: (N, C, H, W) -> Expected format: (N, H, W, C)
    X_train = X_train_tensor.numpy().transpose(0, 2, 3, 1)
    X_test = X_test_tensor.numpy().transpose(0, 2, 3, 1)
    
    # Convert from float [0,1] to uint8 [0,255]
    X_train = (X_train * 255).astype(np.uint8)
    X_test = (X_test * 255).astype(np.uint8)
    
    # Convert labels to numpy
    y_train = y_train.numpy()
    y_test = y_test.numpy()
    
    print(f"CIFAR-10 dataset loaded:")
    print(f"  Training: {X_train.shape} images, {y_train.shape} labels")
    print(f"  Test: {X_test.shape} images, {y_test.shape} labels")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # Save as npz file
    np.savez('datasets/cifar.npz', 
             X_train=X_train, 
             y_train=y_train,
             X_test=X_test, 
             y_test=y_test)
    
    print(f"Saved to: datasets/cifar.npz")
    print("Dataset ready for training!")

if __name__ == "__main__":
    download_cifar10()
