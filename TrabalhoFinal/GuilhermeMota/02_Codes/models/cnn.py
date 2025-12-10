import torch

from torch import nn
import torch.nn.functional as F



class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_size, channel1, channel2, num_l1, num_l2):
        super().__init__()

        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(3, channel1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, in_size, in_size)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            n_features = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(n_features, num_l1)
        self.fc2 = nn.Linear(num_l1, num_l2)
        self.fc3 = nn.Linear(num_l2, 1)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x