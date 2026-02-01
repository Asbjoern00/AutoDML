import torch
import torch.nn as nn
from RieszNet.utilities import make_sequential
import numpy as np


class DOPEDerivativeRieszNetwork(nn.Module):

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
        final_hidden_shared: int = 64,
    ):
        super().__init__()

        self.functional = functional

        # Shared trunk
        self.shared = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=final_hidden_shared,
            n_hidden=n_shared_layers - 1,
            activate_all=True,
        )
        self.rr_head = make_sequential(
            in_dim=final_hidden_shared+1, hidden_dim=n_riesz_weights, out_dim=1, n_hidden=n_riesz_layers
        )
        self.regression = make_sequential(
            in_dim=final_hidden_shared+1, hidden_dim=n_regression_weights, out_dim=1, n_hidden=n_regression_layers
        )

    def _evaluate_regression(self, data):
        t = data.treatments_tensor
        z = self._forward_shared(data)
        tz = torch.concatenate([t, z], dim=1)
        regression_prediction = self.regression(tz)

        return regression_prediction

    def get_riesz_representer(self, data):
        t = data.treatments_tensor
        z = self._forward_shared(data)
        tz = torch.concatenate([t, z], dim=1)
        rr_prediction = self.rr_head(tz)
        return rr_prediction

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

    def _forward_shared(self, data):
        return self.shared(data.covariates_tensor)

    def forward(self, data):
        # Functional applied to RR
        rr_functional = self.functional(data, self.get_riesz_representer)
        rr_output = self.get_riesz_representer(data)
        outcome_prediction = self._evaluate_regression(data)

        return rr_output, rr_functional, outcome_prediction
