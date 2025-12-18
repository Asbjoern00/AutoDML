import torch
import torch.nn.functional as F

from ihdp_average_treatment_effect.dragon_net.dragon_net import DragonNet


class DragonNetWrapper:
    def __init__(self, model, optimizer, l2_lambda):
        self.model = model
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda

    @classmethod
    def create(cls, l2_lambda=1e-2):
        model = DragonNet()
        optimizer = torch.optim.Adam(model.parameters())
        return cls(model, optimizer, l2_lambda)

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
            loss = self._dragon_net_loss(model_output=output, outcomes=outcomes, treatments=treatments) + l2_loss
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

    @staticmethod
    def _dragon_net_loss(
        model_output, outcomes, treatments, outcome_mse_weight=1, treatment_cross_entropy_weight=0.2, tmle_weight=1
    ):
        outcome_mse = F.mse_loss(model_output["base_outcome_prediction"], outcomes)
        treatment_cross_entropy = F.binary_cross_entropy(model_output["treatment_prediction"], treatments)
        tmle_loss = F.mse_loss(model_output["targeted_outcome_prediction"], outcomes)
        return (
            outcome_mse * outcome_mse_weight
            + treatment_cross_entropy * treatment_cross_entropy_weight
            + tmle_loss * tmle_weight
        )
