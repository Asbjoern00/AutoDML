import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from average_treatment_effect.dataset import Dataset as ATEDataset


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

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.model.apply(weight_reset)

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

        y = self.regression_layer1(z)
        y = F.elu(y)
        y = self.regression_layer2(y)
        y = F.elu(y)
        y = self.regression_output(y)

        outcome_prediction = y

        return rr_output, rr_functional, outcome_prediction, self.epsilon


class BaseRieszNetLoss(nn.Module):
    def __init__(self, rr_weight=0.1, tmle_weight=0.45):
        super(BaseRieszNetLoss, self).__init__()
        self.rr_weight = rr_weight
        self.tmle_weight = tmle_weight

    def forward(self, rr_output, rr_functional, outcome_prediction, outcome, epsilon):
        mse = F.mse_loss(outcome_prediction, outcome)
        tmle_loss = F.mse_loss(outcome - outcome_prediction, epsilon * rr_output)
        rr_loss = torch.mean(rr_output**2) - 2 * torch.mean(rr_functional)
        loss = tmle_loss * self.tmle_weight + rr_loss * self.rr_weight + mse * (1 - self.tmle_weight - self.rr_weight)
        # print(rr_loss, tmle_loss, mse)
        return loss


def ate_functional(data, evaluator, treatment_index=0):
    x = data.clone()

    # Predict outcome under treatment T=1
    x_treated = x.clone()
    x_treated[:, treatment_index] = 1
    y1 = evaluator(x_treated)

    # Predict outcome under treatment T=0
    x_control = x.clone()
    x_control[:, treatment_index] = 0
    y0 = evaluator(x_control)

    return y1 - y0


class RieszNetBaseModule:
    def __init__(self, functional, weight_decay=1e-3, rr_weight=0.1, tmle_weight=0.33, epochs=4000):
        self.optimizer = None
        self.model = None
        self.weight_decay = None
        self.functional = functional
        self.set_model(weight_decay)
        self.criterion = BaseRieszNetLoss(rr_weight=rr_weight, tmle_weight=tmle_weight)
        self.epochs = epochs

    def set_model(self, weight_decay):
        self.model = BaseRieszNet(self.functional)
        self.weight_decay = weight_decay
        optimizer_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if n != "epsilon"],
                "weight_decay": self.weight_decay,
            },
            {"params": [self.model.epsilon], "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params, lr=1e-4)

    def _split_into_train_test(self, data, num_folds = 10):
        data.split_into_folds(num_folds)
        train_data = data.get_folds(np.arange(start=2, stop=num_folds+1))
        val_data = data.get_folds([1])
        return train_data, val_data

    def _format_data(self, data):
        if isinstance(data, ATEDataset):
            predictors = torch.concat((data.get_as_tensor("treatments"), data.get_as_tensor("covariates")), dim=1)
            outcomes = data.get_as_tensor("outcomes")
            return predictors, outcomes
        else:
            raise ValueError("Expected a Dataset class. got {}".format(type(data)))

    def train(self, data, patience=50):
        epochs = self.epochs
        train_data, val_data = self._split_into_train_test(data)
        predictors, outcomes = self._format_data(train_data)
        val_predictors, val_outcomes = self._format_data(val_data)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            rr_output, rr_functional, outcome_prediction, epsilon = self.model(predictors)
            loss = self.criterion(rr_output, rr_functional, outcome_prediction, outcomes, epsilon)
            loss.backward()
            if epoch % 200 == 0:
                print(epoch, loss.item())
            self.optimizer.step()

            with torch.no_grad():
                _, _, val_outcome_prediction, _ = self.model(val_predictors)
                val_loss = F.mse_loss(val_outcome_prediction, val_outcomes).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        if best_model_state:
            self.model.load_state_dict(best_model_state)

    def tune_weight_decay(self, data):
        train_data,test_data = self._split_into_train_test(data)
        test_predictors, test_outcomes = self._format_data(test_data)

        weight_decay_grid = [0.0, 1e-4, 1e-3, 1e-2]
        test_losses = np.zeros(len(weight_decay_grid))

        for i in range(len(weight_decay_grid)):
            self.set_model(weight_decay_grid[i])
            self.train(data=train_data)
            rr_output, rr_functional, outcome_prediction, epsilon = self.model(test_predictors)
            test_losses[i] = F.mse_loss(outcome_prediction, test_outcomes).item()

        self.set_model(weight_decay_grid[np.argmin(test_losses)])

    def _regression_function(self, x):
        return self.model(x)[2]

    def get_estimate(self, data):
        dat = torch.concat((data.get_as_tensor("treatments"), data.get_as_tensor("covariates")), dim=1)
        outcomes = data.get_as_tensor("outcomes")
        rr_output, _, outcome_prediction, epsilon = self.model(dat)
        plugin_estimate = torch.mean(self.model.functional(dat, self._regression_function)).item()
        residual = outcomes - outcome_prediction
        one_step_estimate = plugin_estimate + torch.mean(residual * rr_output).item()
        return {"plugin": plugin_estimate, "one step estimate": one_step_estimate}
