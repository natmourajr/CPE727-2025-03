import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from simple_mlp import SimpleMLP
from flexible_mlp import FlexibleMLP


def test_simple_mlp():
    """Test SimpleMLP model"""
    input_size = 11  # Wine quality features
    hidden_size = 64
    output_size = 1

    model = SimpleMLP(input_size, hidden_size, output_size)

    # Test forward pass
    dummy_input = torch.randn(32, input_size)
    output = model(dummy_input)

    assert output.shape == (32, 1), f"Expected shape (32, 1), got {output.shape}"

    # Test backward pass
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dummy_target = torch.randn(32, output_size)

    loss = criterion(output, dummy_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 833, f"Expected 833 parameters, got {total_params}"


def test_flexible_mlp():
    """Test FlexibleMLP model"""
    input_size = 11
    hidden_sizes = [64, 32, 16]
    output_size = 1

    model = FlexibleMLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)

    # Test forward pass
    dummy_input = torch.randn(32, input_size)
    output = model(dummy_input)

    assert output.shape == (32, 1), f"Expected shape (32, 1), got {output.shape}"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3393, f"Expected 3393 parameters, got {total_params}"


def test_flexible_mlp_different_architectures():
    """Test FlexibleMLP with different architectures"""
    # Single hidden layer
    model1 = FlexibleMLP(input_size=10, hidden_sizes=[64], output_size=1)
    output1 = model1(torch.randn(16, 10))
    assert output1.shape == (16, 1)

    # Deep network
    model2 = FlexibleMLP(input_size=10, hidden_sizes=[128, 64, 32, 16, 8], output_size=1)
    output2 = model2(torch.randn(16, 10))
    assert output2.shape == (16, 1)


