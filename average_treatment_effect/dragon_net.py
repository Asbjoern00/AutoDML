import torch
import torch.nn as nn
import torch.nn.functional as F


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
