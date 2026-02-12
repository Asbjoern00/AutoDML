import torch
import torch.nn as nn
from RieszNet.utilities import make_sequential
import numpy as np


class ATERieszNetwork(nn.Module):
    def __init__(
        self,
        functional,
        features_in: int = 26,
        hidden_shared: int = 64,
        n_shared_layers: int = 3,
        n_regression_weights: int = 64,
        n_riesz_weights: int = 64,
        n_regression_layers: int = 2,
        n_riesz_layers: int = 1,
    ):
        super().__init__()

        self.functional = functional

        # Shared trunk
        self.shared = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=hidden_shared,
            n_hidden=n_shared_layers - 1,
            activate_all=True,
        )

        self.rr_head = make_sequential(
            in_dim=hidden_shared,
            hidden_dim=n_riesz_weights,
            out_dim=1,
            n_hidden=n_riesz_layers,
        )

        self.treated_head = make_sequential(
            in_dim=hidden_shared,
            hidden_dim=n_regression_weights,
            out_dim=1,
            n_hidden=n_regression_layers,
        )

        self.untreated_head = make_sequential(
            in_dim=hidden_shared,
            hidden_dim=n_regression_weights,
            out_dim=1,
            n_hidden=n_regression_layers,
        )

        self.epsilon = nn.Parameter(torch.zeros(1))

    def _forward_shared(self, data):
        treatments = data.treatments_tensor
        covariates = data.covariates_tensor
        x = torch.cat((treatments, covariates), dim=1)
        return self.shared(x)

    def _evaluate_regression(self, data):
        adjusted_outcome_prediction = self.forward(data)[3]
        return adjusted_outcome_prediction

    def get_riesz_representer(self, data):
        z = self._forward_shared(data)
        rr = self.rr_head(z)
        rr = torch.clip(rr, -10000, 10000)
        return rr

    def get_plugin_estimate(self, data):
        functional = self.get_functional(data)
        return np.mean(functional.detach().numpy())

    def get_residuals(self, data):
        fitted = self._evaluate_regression(data)
        return data.outcomes_tensor - fitted

    def get_functional(self, data):
        return self.functional(data, self._evaluate_regression)

    def get_correction(self, data):
        residuals = self.get_residuals(data)
        rr = self.get_riesz_representer(data)
        return rr * residuals

    def get_double_robust(self, data):
        plugin = self.get_plugin_estimate(data)
        correction = self.get_correction(data)
        return plugin + np.mean(correction.detach().numpy())

    def forward(self, data):
        # Riesz functional
        rr_functional = self.functional(data, self.get_riesz_representer)

        # Shared representation
        z = self._forward_shared(data)

        # Heads
        rr_output = self.get_riesz_representer(data)
        y_treated = self.treated_head(z)
        y_untreated = self.untreated_head(z)

        t = data.treatments_tensor
        outcome_prediction = t * y_treated + (1 - t) * y_untreated
        adjusted_outcome_prediction = outcome_prediction + self.epsilon * rr_output

        return rr_output, rr_functional, outcome_prediction, adjusted_outcome_prediction


class ATERieszNetworkSimple(nn.Module):

    def __init__(
        self,
        functional,
        features_in: int = 26,
        hidden_shared: int = 64,
        n_shared_layers: int = 2,
        n_regression_weights: int = 64,
        n_riesz_weights: int = 64,
        n_regression_layers: int = 1,
        n_riesz_layers: int = 1,
    ):
        super().__init__()

        self.functional = functional

        # Shared trunk
        self.shared = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=hidden_shared,
            n_hidden=n_shared_layers - 1,
            activate_all=True,
        )

        self.rr_head = make_sequential(
            in_dim=hidden_shared, hidden_dim=n_riesz_weights, out_dim=1, n_hidden=n_riesz_layers
        )

        self.regression_head = make_sequential(
            in_dim=hidden_shared, hidden_dim=n_regression_weights, out_dim=1, n_hidden=n_regression_layers
        )

        self.epsilon = nn.Parameter(torch.zeros(1))

    def _forward_shared(self, data):
        return self.shared(data.net_input)

    def _evaluate_regression(self, data):
        adjusted_outcome_prediction = self.forward(data)[3]
        return adjusted_outcome_prediction

    def get_riesz_representer(self, data):
        z = self._forward_shared(data)
        return self.rr_head(z)

    def get_plugin_estimate(self, data):
        functional = self.get_functional(data)
        return np.mean(functional.detach().numpy())

    def get_residuals(self, data):
        fitted = self._evaluate_regression(data)
        return data.outcomes_tensor - fitted

    def get_functional(self, data):
        return self.functional(data, self._evaluate_regression)

    def get_correction(self, data):
        residuals = self.get_residuals(data)
        rr = self.get_riesz_representer(data)
        return rr * residuals

    def get_double_robust(self, data):
        plugin = self.get_plugin_estimate(data)
        correction = self.get_correction(data)
        return plugin + np.mean(correction.detach().numpy())

    def forward(self, data):
        # Functional applied to RR
        #rr_functional = self.functional(data, self.get_riesz_representer)
        data_treated,data_untreated = data.get_counterfactual_datasets()

        z_treated = self._forward_shared(data_treated)
        z_untreated = self._forward_shared(data_untreated)
        rr_functional = self.rr_head(z_treated) - self.rr_head(z_untreated)

        # Shared representation
        z = self._forward_shared(data)

        # Heads
        rr_output = self.rr_head(z)
        outcome_prediction = self.regression_head(z)
        adjusted_outcome_prediction = outcome_prediction + self.epsilon * rr_output

        return rr_output, rr_functional, outcome_prediction, adjusted_outcome_prediction
