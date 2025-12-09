import torch

from torch import nn



class DeepNeuralNetwork(nn.Module):
    def __init__(self, in_features, num_l1, num_l2, dropout_layer=False):
        super().__init__()

        self.flatten = nn.Flatten()

        if dropout_layer:
            self.hidden_stack = nn.Sequential(
                nn.Linear(in_features, num_l1),
                nn.ReLU(),
                nn.Dropout(0.3), 
                nn.Linear(num_l1, num_l2),
                nn.ReLU(),
                nn.Dropout(0.3), 
                nn.Linear(num_l2, 1),
            )
        else:
            self.hidden_stack = nn.Sequential(
                nn.Linear(in_features, num_l1),
                nn.ReLU(),
                nn.Linear(num_l1, num_l2),
                nn.ReLU(),
                nn.Linear(num_l2, 1),
            )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.hidden_stack(x)

        return logits