import torch
from RieszNet.Loss import RieszLoss
import copy


class DOPERieszNetModule:
    def __init__(self, network, regression_optimizer , rr_optimizer):
        self.network = network
        self.regression_optimizer = regression_optimizer
        self.rr_optimizer = rr_optimizer
        self.regression_loss = torch.nn.MSELoss()
        self.rr_loss = RieszLoss()

    def fit(self, data, informed = "regression"):

        train_data, val_data = data.test_train_split(train_proportion=self.regression_optimizer.early_stopping["proportion"])
        if informed == "regression":

            self.fit_regression(train_data, val_data)

            for group in self.regression_optimizer.optim.param_groups:
                for p in group["params"]:
                    p.requires_grad = False

            self.fit_rr(train_data, val_data)

        if informed == "riesz":
            self.fit_rr(train_data, val_data)

            for group in self.rr_optimizer.optim.param_groups:
                for p in group["params"]:
                    p.requires_grad = False

            self.fit_regression(train_data, val_data)

    def fit_regression(self, train_data, val_data):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.regression_optimizer.epochs):
            self.regression_optimizer.optim.zero_grad()
            _, _, outcome_prediction = self.network(train_data)

            loss = self.regression_loss(outcome_prediction, train_data.outcomes_tensor)
            loss.backward()
            self.regression_optimizer.optim.step()

            with torch.no_grad():
                _, _, outcome_prediction_val = self.network(val_data)

                val_loss = self.regression_loss(val_data.outcomes_tensor, outcome_prediction_val).item()

            if self.regression_optimizer.early_stopping["tolerance"] + val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.network.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.regression_optimizer.early_stopping["rounds"]:
                    print(f"early stopping (Regression) at epoch {epoch}, MSE Loss {best_val_loss:.4f}")
                    break

        self.network.load_state_dict(best_state)

    def fit_rr(self,train_data, val_data):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.rr_optimizer.epochs):
            self.rr_optimizer.optim.zero_grad()
            rr_prediction, rr_functional, _ = self.network(train_data)

            loss = self.rr_loss(rr_prediction, rr_functional)
            loss.backward()
            self.rr_optimizer.optim.step()

            with torch.no_grad():
                rr_prediction_val, rr_functional_val, _ = self.network(val_data)

                val_loss = self.rr_loss(rr_prediction_val, rr_functional_val).item()

            if self.rr_optimizer.early_stopping["tolerance"] + val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.network.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.rr_optimizer.early_stopping["rounds"]:
                    print(f"early stopping (Riesz) at epoch {epoch}")
                    break

        self.network.load_state_dict(best_state)

    def get_plugin(self, data):
        return self.network.get_plugin_estimate(data)

    def get_correction(self, data):
        return self.network.get_correction(data).detach().numpy()

    def get_functional(self, data):
        return self.network.get_functional(data).detach().numpy()

    def get_double_robust(self, data):
        return self.network.get_double_robust(data)
