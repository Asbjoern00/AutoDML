import torch
import torch.nn as nn
import torch.nn.functional as F


class DragonNet(nn.Module):
    def __init__(self):
        super(DragonNet, self).__init__()
        self.base_dragonNet = BaseDragonNet()
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, covariates, treatments):
        base_output = self.base_dragonNet(covariates, treatments)
        untreated_outcome_prediction = base_output["untreated_outcome_predictions"] + self.epsilon / (
            1 - base_output["treatment_predictions"]
        )
        treated_outcome_prediction = (
            base_output["treated_outcome_predictions"] + self.epsilon / base_output["treatment_predictions"]
        )
        targeted_outcome_prediction = (
                treatments * treated_outcome_prediction + (1 - treatments) * untreated_outcome_prediction
        )
        return {
            "treatment_predictions": base_output["treatment_predictions"],
            "untreated_outcome_predictions": untreated_outcome_prediction,
            "treated_outcome_predictions": treated_outcome_prediction,
            "outcome_predictions": base_output["outcome_predictions"],
            "targeted_outcome_predictions": targeted_outcome_prediction,
        }


class BaseDragonNet(nn.Module):
    def __init__(self):
        super(BaseDragonNet, self).__init__()
        self.shared_net = SharedNet()
        self.treated_outcome_net = OutcomeNet()
        self.untreated_outcome_net = OutcomeNet()
        self.propensity_output_layer = nn.Linear(200, 1)

    def forward(self, covariates, treatments):
        shared_state = self.shared_net(covariates)
        propensity_logit = self.propensity_output_layer(shared_state)
        treatment_prediction = torch.sigmoid(propensity_logit)
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


class SharedNet(nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()
        self.layer1 = nn.Linear(25, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 200)

    def forward(self, covariates):
        x = self.layer1(covariates)
        x = F.elu(x)
        x = self.layer2(x)
        x = F.elu(x)
        x = self.layer3(x)
        x = F.elu(x)

        return x


class DragonNetLoss(nn.Module):
    def __init__(self, cross_entropy_weight=0.1, tmle_weight=0.9):
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
