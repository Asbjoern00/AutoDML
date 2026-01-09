import torch
import torch.nn as nn
from RieszNet.utilities import make_sequential
import numpy as np


class DOPEATERieszNetwork(nn.Module):
    def __init__(
        self,
        functional,
        features_in: int = 26,
        hidden_shared: int = 100,
        final_shared: int = 3,
        n_shared_layers: int = 3,
        n_riesz_weights: int = 100,
        n_riesz_layers: int = 2
    ):
        super().__init__()

        self.functional = functional

        # Shared treated chunk
        self.shared_treated = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=final_shared,
            n_hidden=n_shared_layers - 1,
        )
        self.regression_treated = nn.Linear(final_shared, 1)

        self.rr_treated = make_sequential(
            in_dim=final_shared,
            hidden_dim=n_riesz_weights,
            out_dim=1,
            n_hidden=n_riesz_layers
        )

        # Shared untreated chunk
        self.shared_untreated = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=final_shared,
            n_hidden=n_shared_layers - 1,
        )
        self.regression_untreated = nn.Linear(final_shared, 1)

        self.rr_untreated = make_sequential(
            in_dim=final_shared, hidden_dim=n_riesz_weights, out_dim=1, n_hidden=n_riesz_layers
        )

    def _forward_shared(self, data):
        treatments = data.treatments_tensor
        covariates = data.covariates_tensor
        x = torch.cat((treatments, covariates), dim=1)

        shared_treated = self.shared_treated(x)
        shared_untreated = self.shared_untreated(x)

        return shared_treated, shared_untreated

    def _evaluate_regression(self, data):
        outcome_prediction = self.forward(data)[2]
        return outcome_prediction

    def get_riesz_representer(self, data):
        t = data.treatments_tensor
        z_treated, z_untreated = self._forward_shared(data)
        rr_untreated = self.rr_untreated(z_untreated)
        rr_treated = self.rr_treated(z_treated)
        rr_prediction = t * rr_treated + (1 - t) * rr_untreated

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

    def forward(self, data):
        # Riesz functional
        rr_functional = self.functional(data, self.get_riesz_representer)

        # Shared representation
        z_treated, z_untreated = self._forward_shared(data)

        y_treated = self.regression_treated(z_treated)
        y_untreated = self.regression_untreated(z_untreated)

        t = data.treatments_tensor
        outcome_prediction = t * y_treated + (1 - t) * y_untreated

        rr_treated = self.rr_treated(z_treated)
        rr_untreated = self.rr_untreated(z_untreated)
        rr_prediction = t * rr_treated + (1 - t) * rr_untreated

        return rr_prediction, rr_functional, outcome_prediction
