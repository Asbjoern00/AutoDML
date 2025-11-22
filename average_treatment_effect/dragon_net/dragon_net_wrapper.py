import torch

from average_treatment_effect.dragon_net import dragon_net


class DragonNetWrapper:
    def __init__(self, model, criterion, optimizer, l2_lambda):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

    @classmethod
    def create(cls, l2_lambda=1e-4):
        model = dragon_net.DragonNet()
        criterion = dragon_net.DragonNetLoss()
        optimizer = torch.optim.Adam(model.parameters())
        return cls(model, criterion, optimizer, l2_lambda)

    def train(self, data, epochs=500):
        self.model.train()
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        outcomes = data.get_as_tensor("outcomes")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(covariates=covariates, treatments=treatments)
            l2_loss = (
                sum(
                    (param**2).sum()
                    for name, param in self.model.named_parameters()
                    if param.requires_grad and name != "epsilon"
                )
                * self.l2_lambda
            )
            dragon_net_loss = self.criterion(model_output=output, outcomes=outcomes, treatments=treatments)
            loss = dragon_net_loss + l2_loss
            loss.backward()
            self.optimizer.step()

    def evaluate(self, data):
        self.model.eval()
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        outcomes = data.get_as_tensor("outcomes")
        output = self.model(covariates=covariates, treatments=treatments)
        loss = self.criterion(model_output=output, outcomes=outcomes, treatments=treatments)
        return loss.item()

    def get_average_treatment_effect(self, data):
        self.model.eval()
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        with torch.no_grad():
            output = self.model(covariates=covariates, treatments=treatments)
            estimate = torch.mean(output["q1"] - output["q0"])
        return estimate.item()
