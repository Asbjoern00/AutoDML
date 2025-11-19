import torch
from torch import nn


def train_regression_net(data, epochs=100):
    covariates = data.get_as_tensor("covariates")
    treatments = data.get_as_tensor("treatments")
    outcomes = data.get_as_tensor("outcomes")

    model = NaiveNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(covariates, treatments)
        loss = criterion(predictions, outcomes)
        loss.backward()
        optimizer.step()

    return model


class NaiveNet(nn.Module):
    def __init__(self):
        super(NaiveNet, self).__init__()
        self.hidden_layer_1 = nn.Linear(26, 32)
        self.activation_1 = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(32, 16)
        self.activation_2 = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)

    def forward(self, covariates, treatments):
        x = torch.cat([covariates, treatments], dim=1)
        x = self.hidden_layer_1(x)
        x = self.activation_1(x)
        x = self.hidden_layer_2(x)
        x = self.activation_2(x)
        x = self.output_layer(x)
        return x
