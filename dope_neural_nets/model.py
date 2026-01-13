import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, hidden_size=64):
        super(Model, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(11, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.outcome_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.riesz_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def predict_outcome(self, x):
        x = self.shared_layers(x)
        return self.outcome_layers(x)

    def predict_riesz(self, x):
        x = self.shared_layers(x)
        return self.riesz_layers(x)
