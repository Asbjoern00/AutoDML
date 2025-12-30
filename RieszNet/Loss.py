import torch
import torch.nn as nn
import torch.nn.functional as F


class RieszNetLoss(nn.Module):
    def __init__(self, rr_weight=0.1, tmle_weight=1.0, outcome_mse_weight=1.0):
        super(RieszNetLoss, self).__init__()
        self.rr_weight = rr_weight
        self.tmle_weight = tmle_weight
        self.outcome_mse_weight = outcome_mse_weight

    def forward(self, rr_output, rr_functional, outcome_prediction, adjusted_outcome_prediction, outcome):
        mse = F.mse_loss(outcome_prediction, outcome)
        tmle_loss = F.mse_loss(adjusted_outcome_prediction,outcome)
        rr_loss = torch.mean(rr_output**2) - 2 * torch.mean(rr_functional)
        loss = tmle_loss * self.tmle_weight + rr_loss * self.rr_weight + mse * self.outcome_mse_weight
        return loss

class DragonNetLoss(nn.Module):
    def __init__(self, rr_weight=0.1, tmle_weight=1.0, outcome_mse_weight=1.0):
        super(DragonNetLoss, self).__init__()
        self.rr_weight = rr_weight
        self.tmle_weight = tmle_weight
        self.outcome_mse_weight = outcome_mse_weight

    def forward(self, treatment_prediction, treatment, outcome_prediction, adjusted_outcome_prediction, outcome):
        mse = F.mse_loss(outcome_prediction, outcome)
        tmle_loss = F.mse_loss(adjusted_outcome_prediction,outcome)
        rr_loss = F.binary_cross_entropy(treatment_prediction,treatment)
        loss = tmle_loss * self.tmle_weight + rr_loss * self.rr_weight + mse * self.outcome_mse_weight
        return loss