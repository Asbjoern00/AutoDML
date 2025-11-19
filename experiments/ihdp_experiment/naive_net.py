import torch
from torch import nn


class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()
        self.hidden_layer_1 = nn.Linear(26, 32)
        self.activation_1 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(32, 16)
        self.activation_2 = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)

    def forward(self, covariates, treatments):
        x = torch.cat([covariates, treatments], dim=1)
        x = self.hidden_layer_1(x)
        x = self.activation_1(x)
        x = self.hidden_layer_2(x)
        x = self.activation_2(x)
        x = self.output_layer(x)
        return x
