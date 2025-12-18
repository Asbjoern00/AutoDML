import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, covariates, treatments):
        shared_state = self.shared_layers(covariates)
        treatment_prediction = self.treatment_prediction_layer(shared_state).clamp(1e-3, 1 - 1e-3)
        q0 = self.q0_layers(shared_state)
        q1 = self.q1_layers(shared_state)
        base_outcome_prediction = (1 - treatments) * q0 + treatments * q1
        q0 = q0 - self.epsilon / (1 - treatment_prediction)
        q1 = q1 + self.epsilon / treatment_prediction
        targeted_outcome_prediction = (1 - treatments) * q0 + treatments * q1
        return {
            "treatment_prediction": treatment_prediction,
            "q0": q0,
            "q1": q1,
            "base_outcome_prediction": base_outcome_prediction,
            "targeted_outcome_prediction": targeted_outcome_prediction,
        }
