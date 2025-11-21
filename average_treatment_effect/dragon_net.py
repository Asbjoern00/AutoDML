import torch
import torch.nn as nn
import torch.nn.functional as F


class DragonNet(nn.Module):
    def __init__(self):
        super(DragonNet, self).__init__()
        self.base_dragonNet = BaseDragonNet()
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, covariates, treatments):
        output = self.base_dragonNet(covariates, treatments)
        treatment_predictions = output["treatment_predictions"]
        q0 = output["untreated_outcome_predictions"]
        q1 = output["treated_outcome_predictions"]

        q0 = q0 + self.epsilon / (1 - treatment_predictions)
        q1 = q1 + self.epsilon / treatment_predictions
        targeted_outcome_prediction = treatments * q1 + (1 - treatments) * q0

        return {
            "treatment_predictions": treatment_predictions,
            "untreated_outcome_predictions": q0,
            "treated_outcome_predictions": q1,
            "outcome_predictions": output["outcome_predictions"],
            "targeted_outcome_predictions": targeted_outcome_prediction,
        }


class BaseDragonNet(nn.Module):
    def __init__(self):
        super(BaseDragonNet, self).__init__()
        self.shared_layer1 = nn.Linear(25, 200)
        self.shared_layer2 = nn.Linear(200, 200)
        self.shared_layer3 = nn.Linear(200, 200)

        self.treated_outcome_net = OutcomeNet()
        self.untreated_outcome_net = OutcomeNet()
        self.treatment_prediction_layer = nn.Linear(200, 1)

    def forward(self, covariates, treatments):
        shared_state = self.shared_layer1(covariates)
        shared_state = F.elu(shared_state)
        shared_state = self.shared_layer2(shared_state)
        shared_state = F.elu(shared_state)
        shared_state = self.shared_layer3(shared_state)
        shared_state = F.elu(shared_state)

        treatment_prediction = self.treatment_prediction_layer(shared_state).sigmoid().clamp(1e-3, 1 - 1e-3)
        untreated_outcome_prediction = self.untreated_outcome_net(shared_state)
        treated_outcome_prediction = self.treated_outcome_net(shared_state)
        outcome_prediction = treatments * treated_outcome_prediction + (1 - treatments) * untreated_outcome_prediction

        return {
            "treatment_predictions": treatment_prediction,
            "untreated_outcome_predictions": untreated_outcome_prediction,
            "treated_outcome_predictions": treated_outcome_prediction,
            "outcome_predictions": outcome_prediction,
        }


class OutcomeNet(nn.Module):
    def __init__(self):
        super(OutcomeNet, self).__init__()
        self.layer1 = nn.Linear(200, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 1)

    def forward(self, shared_state):
        x = self.layer1(shared_state)
        x = F.elu(x)
        x = self.layer2(x)
        x = F.elu(x)
        x = self.layer3(x)
        return x


class DragonNetLoss(nn.Module):
    def __init__(self, cross_entropy_weight=0.1, tmle_weight=0.5):
        super(DragonNetLoss, self).__init__()
        self.base_loss = BaseDragonNetLoss(cross_entropy_weight)
        self.tmle_weight = tmle_weight

    def forward(self, model_output, treatments, outcomes):
        base_loss = self.base_loss(model_output, treatments, outcomes)
        tmle_loss = F.mse_loss(model_output["targeted_outcome_predictions"], outcomes)
        return self.tmle_weight * tmle_loss + (1 - self.tmle_weight) * base_loss


class BaseDragonNetLoss(nn.Module):
    def __init__(self, cross_entropy_weight=0.1):
        super(BaseDragonNetLoss, self).__init__()
        self.cross_entropy_weight = cross_entropy_weight

    def forward(self, model_output, treatments, outcomes):
        treatment_cross_entropy = F.binary_cross_entropy(model_output["treatment_predictions"], treatments)
        outcome_mse = F.mse_loss(model_output["outcome_predictions"], outcomes)
        return self.cross_entropy_weight * treatment_cross_entropy + (1 - self.cross_entropy_weight) * outcome_mse
