import torch
import torch.nn.functional as F
import numpy as np
from average_treatment_effect.dataset import Dataset as ATEDataset
from average_treatment_effect.riesz_net.RieszNetBase import BiHeadedBaseRieszNet, BaseRieszNet
from average_treatment_effect.riesz_net.BaseRieszNetLoss import BaseRieszNetLoss
from AveragePartialDerivative.dataset import Dataset as DerivativeDataset


class RieszNetBaseModule:
    def __init__(
        self, functional, weight_decay=1e-2, rr_weight=0.1, tmle_weight=1.0, outcome_mse_weight=1.0, epochs=2500, biheaded = False,
            in_features = 26
    ):
        self.optimizer = None
        self.model = None
        self.weight_decay = None
        self.biheaded = biheaded
        self.functional = functional
        self.in_features = in_features
        self.set_model(weight_decay)
        self.criterion = BaseRieszNetLoss(
            rr_weight=rr_weight, tmle_weight=tmle_weight, outcome_mse_weight=outcome_mse_weight
        )
        self.epochs = epochs

    def set_model(self, weight_decay):
        if self.biheaded:
            self.model = BiHeadedBaseRieszNet(self.functional,self.in_features)
        else:
            self.model = BaseRieszNet(self.functional,self.in_features)

        self.weight_decay = weight_decay
        optimizer_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if n != "epsilon"],
                "weight_decay": self.weight_decay,
            },
            {"params": [self.model.epsilon], "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params, lr=1e-4)

    @staticmethod
    def _split_into_train_test(data, num_folds=10):
        data.split_into_folds(num_folds)
        train_data = data.get_folds(np.arange(start=2, stop=num_folds + 1))
        val_data = data.get_folds([1])
        return train_data, val_data

    @staticmethod
    def _format_data(data):
        if isinstance(data, ATEDataset) or isinstance(data, DerivativeDataset):
            predictors = torch.concat((data.get_as_tensor("treatments"), data.get_as_tensor("covariates")), dim=1)
            outcomes = data.get_as_tensor("outcomes")
            return predictors, outcomes
        else:
            raise ValueError("Expected a Dataset class. got {}".format(type(data)))

    def _regression_function(self, x):
        return self.model(x)[2]

    def predict(self, data):
        predictors, outcomes = self._format_data(data)
        functional_eval = self.model.functional(predictors, self._regression_function)
        rr_output, _, outcome_prediction, _ = self.model(predictors)

        return rr_output, functional_eval, outcome_prediction

    def train(self, data, patience=40, delta=1e-3):
        epochs = self.epochs
        train_data, val_data = self._split_into_train_test(data=data)
        predictors, outcomes = self._format_data(data=train_data)
        val_predictors, val_outcomes = self._format_data(data=val_data)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            rr_output, rr_functional, outcome_prediction, epsilon = self.model(predictors)
            loss = self.criterion(rr_output, rr_functional, outcome_prediction, outcomes, epsilon)
            loss.backward()
            self.optimizer.step()

            rr_output_val, rr_functional_val, outcome_prediction_val, val_epsilon = self.model(val_predictors)
            val_loss = self.criterion(
                rr_output_val, rr_functional_val, outcome_prediction_val, val_outcomes, val_epsilon
            ).item()

            if delta + val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        if best_model_state:
            self.model.load_state_dict(best_model_state)

    def tune_weight_decay(self, data):
        train_data, test_data = self._split_into_train_test(data)
        test_predictors, test_outcomes = self._format_data(test_data)

        weight_decay_grid = [0.0, 1e-4, 1e-3, 1e-2]
        test_losses = np.zeros(len(weight_decay_grid))

        for i in range(len(weight_decay_grid)):
            self.set_model(weight_decay_grid[i])
            self.train(data=train_data)
            rr_output, rr_functional, outcome_prediction, epsilon = self.model(test_predictors)
            test_losses[i] = F.mse_loss(outcome_prediction, test_outcomes).item()

        self.set_model(weight_decay_grid[np.argmin(test_losses)])

    def get_estimate(self, data):
        dat = torch.concat((data.get_as_tensor("treatments"), data.get_as_tensor("covariates")), dim=1)
        outcomes = data.get_as_tensor("outcomes")
        rr_output, _, outcome_prediction, epsilon = self.model(dat)
        plugin_estimate = torch.mean(self.model.functional(dat, self._regression_function)).item()
        residual = outcomes - outcome_prediction
        one_step_estimate = plugin_estimate + torch.mean(residual * rr_output).item()
        return {"plugin": plugin_estimate, "one step estimate": one_step_estimate}

    def fit(self, data, n_crossfit=1):
        data.split_into_folds(n_crossfit)
        splits = np.arange(1, n_crossfit + 1)
        rr_output, functional_eval, outcome_prediction, outcomes = (
            torch.zeros(data.outcomes.shape[0],1),
            torch.zeros(data.outcomes.shape[0],1),
            torch.zeros(data.outcomes.shape[0],1),
            torch.zeros(data.outcomes.shape[0],1),
        )

        n_cum = 0
        for l in range(n_crossfit):
            self.set_model(weight_decay=self.weight_decay)  # reset parameters of model
            if n_crossfit == 1:
                train_data = data
                test_data = data
            else:
                train_data = data.get_folds(np.delete(splits, l))
                test_data = data.get_folds([splits[l]])

            test_predictors, test_outcomes = self._format_data(data=test_data)
            n_test = test_outcomes.shape[0]

            self.train(data=train_data)
            (
                rr_output[n_cum : n_cum + n_test],
                functional_eval[n_cum : n_cum + n_test],
                outcome_prediction[n_cum : n_cum + n_test],
            ) = self.predict(data=test_data)
            outcomes[n_cum : n_cum + n_test] = test_outcomes
            n_cum += n_test

        plugin_estimate = torch.mean(functional_eval)
        os_estimate = plugin_estimate + torch.mean((outcomes - outcome_prediction) * rr_output)
        variance_estimate = torch.mean(
            (functional_eval + (outcomes - outcome_prediction) * rr_output - plugin_estimate) ** 2
        )
        return {
            "plugin": plugin_estimate.item(),
            "one step estimate": os_estimate.item(),
            "std_error": np.sqrt(variance_estimate.item()/n_cum),
        }
