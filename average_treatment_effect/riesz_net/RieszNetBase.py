import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseRieszNet(nn.Module):
    def __init__(self, functional):
        super(BaseRieszNet, self).__init__()
        self.functional = functional

        self.shared1 = nn.Linear(26, 200)
        self.shared2 = nn.Linear(200, 200)
        self.shared3 = nn.Linear(200, 200)

        self.rrOutput = nn.Linear(200, 1)

        self.regression_layer1 = nn.Linear(200, 100)
        self.regression_layer2 = nn.Linear(100, 100)
        self.regression_output = nn.Linear(100, 1)

        self.epsilon = nn.Parameter(torch.Tensor([0.0]))

    def _forward_shared(self, data):
        z = self.shared1(data)
        z = F.elu(z)
        z = self.shared2(z)
        z = F.elu(z)
        z = self.shared3(z)
        z = F.elu(z)
        return z

    def _evaluate_riesz(self, data):
        z = self._forward_shared(data)
        return self.rrOutput(z)

    def forward(self, data):
        rr_functional = self.functional(data, self._evaluate_riesz)

        z = self._forward_shared(data)
        rr_output = self.rrOutput(z)

        y = self.treated_regression_layer1(z)
        y = F.elu(y)
        y = self.treated_regression_layer2(y)
        y = F.elu(y)
        y = self.treated_regression_output(y)

        outcome_prediction = y

        return rr_output, rr_functional, outcome_prediction, self.epsilon


class BiHeadedBaseRieszNet(nn.Module):
    def __init__(self, functional):
        super(BiHeadedBaseRieszNet, self).__init__()
        self.functional = functional

        self.shared1 = nn.Linear(26, 200)
        self.shared2 = nn.Linear(200, 200)
        self.shared3 = nn.Linear(200, 200)

        self.rrOutput = nn.Linear(200, 1)

        self.untreated_regression_layer1 = nn.Linear(200, 100)
        self.untreated_regression_layer2 = nn.Linear(100, 100)
        self.untreated_regression_output = nn.Linear(100, 1)

        self.treated_regression_layer1 = nn.Linear(200, 100)
        self.treated_regression_layer2 = nn.Linear(100, 100)
        self.treated_regression_output = nn.Linear(100, 1)

        self.epsilon = nn.Parameter(torch.Tensor([0.0]))

    def _forward_shared(self, data):
        z = self.shared1(data)
        z = F.elu(z)
        z = self.shared2(z)
        z = F.elu(z)
        z = self.shared3(z)
        z = F.elu(z)
        return z

    def _evaluate_riesz(self, data):
        z = self._forward_shared(data)
        return self.rrOutput(z)

    def forward(self, data):
        rr_functional = self.functional(data, self._evaluate_riesz)

        z = self._forward_shared(data)
        rr_output = self.rrOutput(z)

        y_treated = self.treated_regression_layer1(z)
        y_treated = F.elu(y_treated)
        y_treated = self.treated_regression_layer2(y_treated)
        y_treated = F.elu(y_treated)
        y_treated = self.treated_regression_output(y_treated)

        y_untreated = self.untreated_regression_layer1(z)
        y_untreated = F.elu(y_untreated)
        y_untreated = self.untreated_regression_layer2(y_untreated)
        y_untreated = F.elu(y_untreated)
        y_untreated = self.untreated_regression_output(y_untreated)

        outcome_prediction = y_treated*data[:,[0]] + y_untreated*(1-data[:,[0]])

        return rr_output, rr_functional, outcome_prediction, self.epsilon
