import torch
import torch.nn as nn
import torch.nn.functional as F


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
