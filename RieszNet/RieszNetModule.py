import torch
import numpy as np


class RieszNetModule:
    def __init__(self, network, optimizer, loss):
        self.network = network
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, data):

        train_data, val_data = data.test_train_split(train_proportion=self.optimizer.early_stopping["proportion"])

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.optimizer.epochs):
            self.optimizer.optim.zero_grad()
            rr_output, rr_functional, outcome_prediction, adjusted_outcome_prediction = self.network(train_data)
            loss = self.loss(
                rr_output, rr_functional, outcome_prediction, adjusted_outcome_prediction, train_data.outcomes_tensor
            )
            loss.backward()
            self.optimizer.optim.step()

            with torch.no_grad():
                rr_output_val, rr_functional_val, outcome_prediction_val, adjusted_outcome_prediction_val = (
                    self.network(val_data)
                )
                val_loss = self.loss(
                    rr_output_val,
                    rr_functional_val,
                    outcome_prediction_val,
                    adjusted_outcome_prediction_val,
                    val_data.outcomes_tensor,
                ).item()

            if self.optimizer.early_stopping["tolerance"] + val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.optimizer.early_stopping["rounds"]:
                    print(f"early stopping at epoch {epoch}")
                    break

    def get_plugin(self, data):
        return self.network.get_plugin_estimate(data)

    def get_correction(self, data):
        return self.network.get_correction(data).detach().numpy()

    def get_functional(self, data):
        return self.network.get_functional(data).detach().numpy()

    def get_double_robust(self, data):
        return self.network.get_double_robust(data)
