import torch.nn as nn
from torchvision.models import resnet18

def get_cnn_model(num_classes=10):
    """
    Returns a modified ResNet-18 model suitable for 32x32 CIFAR-10 images.
    """
    # Initialize ResNet-18 without pre-trained weights
    model = resnet18(weights=None, num_classes=num_classes)
    
    # 1. Modify the first convolutional layer for smaller images (32x32)
    # Original: 7x7 stride 2. Modified: 3x3 stride 1.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove the initial MaxPool layer which drastically reduces resolution
    model.maxpool = nn.Identity() 

    # Note: If you have issues with CUDA/MPS memory, you can use a smaller model 
    # like ResNet9 or a custom 4-layer CNN here instead.
    
    return model