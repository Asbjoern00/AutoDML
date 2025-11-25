import torch.nn.functional as F
import torch.nn as nn
import torch



class BaseRieszNetLoss(nn.Module):
    def __init__(self, rr_weight=0.1, tmle_weight=1.0, outcome_mse_weight=1.0):
        super(BaseRieszNetLoss, self).__init__()
        self.rr_weight = rr_weight
        self.tmle_weight = tmle_weight
        self.outcome_mse_weight = outcome_mse_weight

    def forward(self, rr_output, rr_functional, outcome_prediction, outcome, epsilon):
        mse = F.mse_loss(outcome_prediction, outcome)
        tmle_loss = F.mse_loss(outcome - outcome_prediction, epsilon * rr_output)
        rr_loss = torch.mean(rr_output**2) - 2 * torch.mean(rr_functional)
        loss = tmle_loss * self.tmle_weight + rr_loss * self.rr_weight + mse * self.outcome_mse_weight
        return loss