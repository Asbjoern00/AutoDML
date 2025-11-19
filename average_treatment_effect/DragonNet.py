import torch.nn as nn


class DragonNet(nn.Module):
    def __init__(self):
        super(DragonNet, self).__init__()
        self.shared1 = nn.Linear(25, 200)
        self.shared1_activation = nn.ELU()

        self.shared2 = nn.Linear(200, 200)
        self.shared2_activation = nn.ELU()

        self.shared3 = nn.Linear(200, 200)
        self.shared3_activation = nn.ELU()

        self.propensity_output = nn.Linear(200, 1)
        self.propensity_activation = nn.Sigmoid()

        self.treated_layer1 = nn.Linear(200, 100)
        self.treated_layer1_activation = nn.ELU()

        self.treated_layer2 = nn.Linear(100, 100)
        self.treated_layer2_activation = nn.ELU()

        self.treated_output = nn.Linear(100, 1)

        self.untreated_layer1 = nn.Linear(200, 100)
        self.untreated_layer1_activation = nn.ELU()

        self.untreated_layer2 = nn.Linear(100, 100)
        self.untreated_layer2_activation = nn.ELU()

        self.untreated_output = nn.Linear(100, 1)

    def forward(self, covariates, treatment):
        z = self.shared1(covariates)
        z = self.shared1_activation(z)
        z = self.shared2(z)
        z = self.shared2_activation(z)
        z = self.shared3(z)
        z = self.shared3_activation(z)

        propensity_output = self.propensity_output(z)
        propensity_output = self.propensity_activation(propensity_output)

        q1 = self.treated_layer1(z)
        q1 = self.treated_layer1_activation(q1)
        q1 = self.treated_layer2(q1)
        q1 = self.treated_layer2_activation(q1)
        q1 = self.treated_output(q1)

        q0 = self.untreated_layer1(z)
        q0 = self.untreated_layer1_activation(q0)
        q0 = self.untreated_layer2(q0)
        q0 = self.untreated_layer2_activation(q0)
        q0 = self.untreated_output(q0)

        regression_output = treatment * q1 + (1 - treatment) * q0

        return propensity_output, q0, q1, regression_output


class DragonNetLoss(nn.Module):
    def __init__(self, cross_entropy_weight=0.5):
        super(DragonNetLoss, self).__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.mse = nn.MSELoss()
        self.cross = nn.CrossEntropyLoss()

    def forward(self, treatment_prediction, treatment, outcome_prediction, outcome):
        mse = self.mse(outcome_prediction, outcome)
        cross = self.cross(treatment_prediction, treatment)
        loss = mse * (1 - self.cross_entropy_weight) + cross * self.cross_entropy_weight
        return loss



