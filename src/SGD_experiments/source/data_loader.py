import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def get_cifar10_data(batch_size=128):
    # Mean and standard deviation for CIFAR-10 (used for normalization)
    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    # 1. Training Transforms (with standard augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 2. Validation/Test Transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load dataset
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Split training set into train and validation (50k total, splitting to 45k/5k)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    
    # Set generator for reproducibility during split
    g = torch.Generator().manual_seed(42) 
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size], generator=g)

    # Create DataLoaders
    # Pin_memory=True and num_workers may need adjustment based on your M4 memory/cores
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, valloader, testloader