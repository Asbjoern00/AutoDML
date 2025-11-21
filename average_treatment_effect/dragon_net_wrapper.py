import torch

from average_treatment_effect import dragon_net


class DragonNetWrapper:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    @classmethod
    def create_base_dragon_net(cls, weight_decay = 0, learning_rate = 1e-3):
        model = dragon_net.BaseDragonNet()
        criterion = dragon_net.BaseDragonNetLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        return cls(model, criterion, optimizer)

    @classmethod
    def create_dragon_net(cls, weight_decay = 0, learning_rate = 1e-3):
        model = dragon_net.DragonNet()
        criterion = dragon_net.DragonNetLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        return cls(model, criterion, optimizer)

    def train_model(self, data, epochs=1000):
        self.model.train()
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        outcomes = data.get_as_tensor("outcomes")
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            model_output = self.model(covariates=covariates, treatments=treatments)
            loss = self.criterion(model_output=model_output, treatments=treatments, outcomes=outcomes)
            loss.backward()
            self.optimizer.step()

    def evaluate(self, data):
        self.model.eval()
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        outcomes = data.get_as_tensor("outcomes")
        with torch.no_grad():
            model_output = self.model(covariates=covariates, treatments=treatments)
            loss = self.criterion(model_output=model_output, treatments=treatments, outcomes=outcomes)
        return loss.item()

    def get_average_treatment_effect(self, data):
        self.model.eval()
        covariates = data.get_as_tensor("covariates")
        treatments = data.get_as_tensor("treatments")
        outcomes = data.get_as_tensor("outcomes")
        with torch.no_grad():
            model_output = self.model(covariates=covariates, treatments=treatments)
            plugin_estimate = torch.mean(
                model_output["treated_outcome_predictions"] - model_output["untreated_outcome_predictions"]
            )
            riesz_representer = treatments * model_output["treatment_predictions"] + (1 - treatments) * (
                1 - model_output["treatment_predictions"]
            )
            residual = outcomes - model_output["outcome_predictions"]
            one_step_estimate = plugin_estimate + torch.mean(residual * riesz_representer)
            return {"plugin_estimate": plugin_estimate.item(), "one_step_estimate": one_step_estimate.item()}
