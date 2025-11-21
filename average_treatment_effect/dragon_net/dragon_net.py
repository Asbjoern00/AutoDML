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
        treatment_prediction = self.treatment_prediction_layer(treatments)
        q0 = self.q0_layers(shared_state)
        q1 = self.q1_layers(treatment_prediction)
        base_outcome_prediction = (1 - treatments) * q0 + treatments * q1
        q0 = q0 + self.epsilon / treatment_prediction
        q1 = q1 + self.epsilon / (1 - treatment_prediction)
        targeted_outcome_prediction = (1 - treatments) * q0 + treatments * q1
        return {
            "treatment_prediction": treatment_prediction,
            "q0": q0,
            "q1": q1,
            "base_outcome_prediction": base_outcome_prediction,
            "targeted_outcome_prediction": targeted_outcome_prediction,
        }


class DragonNetLoss(nn.Module):
    def __init__(self, outcome_mse_weight=1, treatment_cross_entropy_weight=0.2, tmle_weight=1):
        super(DragonNetLoss, self).__init__()
        self.outcome_mse_weight = outcome_mse_weight
        self.treatment_cross_entropy_weight = treatment_cross_entropy_weight
        self.tmle_weight = tmle_weight

    def forward(self, model_output, outcomes, treatments):
        outcome_mse = F.mse_loss(model_output["base_outcome_prediction"], outcomes)
        treatment_cross_entropy = F.binary_cross_entropy(model_output["treatment_prediction"], treatments)
        tmle_loss = F.mse_loss(model_output["targeted_outcome_prediction"], outcomes)
        return (
            outcome_mse * self.outcome_mse_weight
            + treatment_cross_entropy * self.treatment_cross_entropy_weight
            + tmle_loss * self.tmle_weight
        )
