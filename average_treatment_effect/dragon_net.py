import torch
import torch.nn as nn
import torch.nn.functional as F


class DragonNetModule:
    def __init__(self):
        self.model = DragonNet()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = DragonNetLoss()

    def train(self, epochs, data):
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        outcomes = data.get_as_tensor("outcomes")

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            treatment_predictions, _, _, outcome_predictions = self.model(covariates, treatments)
            loss = self.criterion(treatment_predictions, treatments, outcome_predictions, outcomes)
            loss.backward()
            self.optimizer.step()

    def get_average_treatment_effect_estimate(self, data):
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        outcomes = data.get_as_tensor("outcomes")
        treatment_predictions, q0, q1, outcome_predictions = self.model(covariates, treatments)
        plugin_estimate = torch.mean(q1 - q0).item()
        residual = outcomes - outcome_predictions
        riesz_representer = treatments / treatment_predictions - (1 - treatments) / (1 - treatment_predictions)
        one_step_estimate = plugin_estimate + torch.mean(residual * riesz_representer).item()
        return {"plugin": plugin_estimate, "one step estimate": one_step_estimate}


class DragonNet(nn.Module):
    def __init__(self):
        super(DragonNet, self).__init__()
        self.shared1 = nn.Linear(25, 200)
        self.shared2 = nn.Linear(200, 200)
        self.shared3 = nn.Linear(200, 200)

        self.propensity_output = nn.Linear(200, 1)

        self.treated_layer1 = nn.Linear(200, 100)
        self.treated_layer2 = nn.Linear(100, 100)
        self.treated_output = nn.Linear(100, 1)

        self.untreated_layer1 = nn.Linear(200, 100)
        self.untreated_layer2 = nn.Linear(100, 100)
        self.untreated_output = nn.Linear(100, 1)

    def forward(self, covariates, treatment):
        z = self.shared1(covariates)
        z = F.elu(z)
        z = self.shared2(z)
        z = F.elu(z)
        z = self.shared3(z)
        z = F.elu(z)

        propensity_output = self.propensity_output(z)
        propensity_output = F.sigmoid(propensity_output)

        q1 = self.treated_layer1(z)
        q1 = F.elu(q1)
        q1 = self.treated_layer2(q1)
        q1 = F.elu(q1)
        q1 = self.treated_output(q1)

        q0 = self.untreated_layer1(z)
        q0 = F.elu(q0)
        q0 = self.untreated_layer2(q0)
        q0 = F.elu(q0)
        q0 = self.untreated_output(q0)

        treatment_prediction = propensity_output
        outcome_prediction = treatment * q1 + (1 - treatment) * q0

        return treatment_prediction, q0, q1, outcome_prediction


class DragonNetLoss(nn.Module):
    def __init__(self, cross_entropy_weight=0.5):
        super(DragonNetLoss, self).__init__()
        self.cross_entropy_weight = cross_entropy_weight

    def forward(self, treatment_prediction, treatment, outcome_prediction, outcome):
        mse = F.mse_loss(outcome_prediction, outcome)
        cross = F.cross_entropy(treatment_prediction, treatment)
        loss = mse * (1 - self.cross_entropy_weight) + cross * self.cross_entropy_weight
        return loss
