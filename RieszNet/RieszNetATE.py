import torch
import torch.nn as nn
import torch.nn.functional as F
from RieszNet.utilities import make_sequential
import numpy as np


class ATERieszNetwork(nn.Module):
    def __init__(
        self,
        functional,
        features_in: int = 26,
        hidden_shared: int = 200,
        n_shared_layers: int = 3,
        n_regression_weights: int = 100,
        n_regression_layers: int = 2,
    ):
        super().__init__()

        self.functional = functional

        # Shared trunk
        self.shared = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=hidden_shared,
            n_hidden=n_shared_layers - 1,
        )

        self.rr_head = nn.Linear(hidden_shared, 1)

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
        # Riesz functional
        rr_functional = self.functional(data, self.get_riesz_representer)

        # Shared representation
        z = self._forward_shared(data)

        # Heads
        rr_output = self.rr_head(z)
        y_treated = self.treated_head(z)
        y_untreated = self.untreated_head(z)

        # Treatment indicator assumed in column 0
        t = data.treatments_tensor
        outcome_prediction = t * y_treated + (1 - t) * y_untreated
        adjusted_outcome_prediction = outcome_prediction + self.epsilon * rr_output

        return rr_output, rr_functional, outcome_prediction, adjusted_outcome_prediction
