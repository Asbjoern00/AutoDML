import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDragonNet(nn.Module):
    def __init__(self):
        super(BaseDragonNet, self).__init__()
        self.shared_net = SharedNet()
        self.treated_outcome_net = OutcomeNet()
        self.untreated_outcome_net = OutcomeNet()
        self.propensity_output_layer = nn.Linear(200, 1)

    def forward(self, covariates, treatment):
        shared_state = self.shared_net(covariates)
        propensity_logit = self.propensity_output_layer(shared_state)
        treatment_prediction = torch.sigmoid(propensity_logit)
        untreated_outcome_prediction = self.untreated_outcome_net(shared_state)
        treated_outcome_prediction = self.treated_outcome_net(shared_state)
        outcome_prediction = treatment * treated_outcome_prediction + (1 - treatment) * untreated_outcome_prediction

        return {
            "treatment_prediction": treatment_prediction,
            "untreated_outcome_prediction": untreated_outcome_prediction,
            "treated_outcome_prediction": treated_outcome_prediction,
            "outcome_prediction": outcome_prediction,
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
