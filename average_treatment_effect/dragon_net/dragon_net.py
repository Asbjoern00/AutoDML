import torch
import torch.nn as nn


class DragonNet(nn.Module):
    def __init__(self):
        super(DragonNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(25, 200), nn.ELU(), nn.Linear(200, 200), nn.ELU(), nn.Linear(200, 200), nn.ELU()
        )
        self.treatment_prediction_layer = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())
        self.q0_layers = nn.Sequential(nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 100), nn.ELU(), nn.Linear(100, 1))
        self.q1_layers = nn.Sequential(nn.Linear(200, 100), nn.ELU(), nn.Linear(100, 100), nn.ELU(), nn.Linear(100, 1))
        self.epsilon = nn.Parameter(torch.zeros(1))
