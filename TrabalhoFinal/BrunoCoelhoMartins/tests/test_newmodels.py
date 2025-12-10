from ThreeWToolkit.models import CNNConfig, CNN
import torch

cfg = CNNConfig(
    input_size=10,
    num_classes=2,
    conv_channels=[16],
    kernel_sizes=[1],
    output_size=32   # depends on the conv output size
)

model = CNN(cfg)

x = torch.randn(4, 1, 10)   # B=4, C=1, L=10
y = model(x)

print("CNN output:", y.shape)

from ThreeWToolkit.models import LSTMConfig, LSTM
import torch

cfg = LSTMConfig(
    input_size=10,       # sequence length
    hidden_size=32,
    num_layers=1,
    output_size=2,       # REQUIRED
    num_classes=2,       # probably redundant if you use output_size
    bidirectional=False,
)

model = LSTM(cfg)

x = torch.randn(4, 1, 10)   # B=4, C=1, L=10
y = model(x)

print("LSTM output:", y.shape)

from ThreeWToolkit.models import GRUConfig, GRU
import torch

cfg = GRUConfig(
    input_size=10,
    hidden_size=32,
    num_layers=1,
    output_size=2,       # REQUIRED
    num_classes=2,
    bidirectional=False,
)

model = GRU(cfg)

x = torch.randn(4, 1, 10)   # B=4, C=1, L=10
y = model(x)

print("GRU output:", y.shape)
