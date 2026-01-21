import torch
import torch.nn as nn
from RieszNet.utilities import make_sequential
import numpy as np


class DOPEATERieszNetworkSep(nn.Module):
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

        # Shared treated chunk
        self.shared_treated = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=final_hidden_shared,
            n_hidden=n_shared_layers - 1,
            activate_all=True,
        )

        self.shared_untreated = make_sequential(
            in_dim=features_in,
            hidden_dim=hidden_shared,
            out_dim=final_hidden_shared,
            n_hidden=n_shared_layers - 1,
            activate_all=True,
        )

        self.regression_treated = make_sequential(
            in_dim=final_hidden_shared, hidden_dim=n_regression_weights, out_dim=1, n_hidden=n_regression_layers
        )

        self.regression_untreated = make_sequential(
            in_dim=final_hidden_shared, hidden_dim=n_regression_weights, out_dim=1, n_hidden=n_regression_layers
        )

        self.riesz_treated = make_sequential(
            in_dim=final_hidden_shared, hidden_dim=n_riesz_weights, out_dim=1, n_hidden=n_riesz_layers
        )

        self.riesz_untreated = make_sequential(
            in_dim=final_hidden_shared, hidden_dim=n_riesz_weights, out_dim=1, n_hidden=n_riesz_layers
        )

    def _forward_shared(self, data):
        z_treated, z_untreated = self.shared_treated(data.covariates_tensor), self.shared_untreated(
            data.covariates_tensor
        )
        return z_treated, z_untreated

    def _evaluate_regression(self, data):
        t = data.treatments_tensor
        z_treated, z_untreated = self._forward_shared(data)

        regression_untreated = self.regression_untreated(z_untreated)
        regression_treated = self.regression_treated(z_treated)
        regression_prediction = t * regression_treated + (1 - t) * regression_untreated

        return regression_prediction

    def get_riesz_representer(self, data):
        t = data.treatments_tensor
        z_treated, z_untreated = self._forward_shared(data)
        rr_untreated = self.riesz_untreated(z_untreated)
        rr_treated = self.riesz_treated(z_treated)
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
        rr_prediction = self.get_riesz_representer(data)
        outcome_prediction = self._evaluate_regression(data)

        return rr_prediction, rr_functional, outcome_prediction


class DOPEATERieszNetworkSimple(nn.Module):

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
            in_dim=final_hidden_shared, hidden_dim=n_riesz_weights, out_dim=1, n_hidden=n_riesz_layers
        )

        self.regression_treated = make_sequential(
            in_dim=final_hidden_shared, hidden_dim=n_regression_weights, out_dim=1, n_hidden=n_regression_layers
        )

        self.regression_untreated = make_sequential(
            in_dim=final_hidden_shared, hidden_dim=n_regression_weights, out_dim=1, n_hidden=n_regression_layers
        )

    def _evaluate_regression(self, data):
        t = data.treatments_tensor
        z = self._forward_shared(data)

        regression_untreated = self.regression_untreated(z)
        regression_treated = self.regression_treated(z)
        regression_prediction = t * regression_treated + (1 - t) * regression_untreated

        return regression_prediction

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

    def _forward_shared(self, data):
        return self.shared(data.net_input)

    def forward(self, data):
        # Functional applied to RR
        rr_functional = self.functional(data, self.get_riesz_representer)
        rr_output = self.get_riesz_representer(data)
        outcome_prediction = self._evaluate_regression(data)

        return rr_output, rr_functional, outcome_prediction


class DOPEATERieszNetworkNonShared(nn.Module):
    def __init__(
        self,
        functional,
        features_in: int = 26,
        n_regression_weights: int = 64,
        n_riesz_weights: int = 64,
        n_regression_layers: int = 3,
        n_riesz_layers: int = 3,
    ):
        super().__init__()

        self.functional = functional

        self.rr = make_sequential(in_dim=features_in, hidden_dim=n_riesz_weights, out_dim=1, n_hidden=n_riesz_layers)

        self.shared_regression = make_sequential(
            in_dim=features_in - 1, hidden_dim=n_regression_weights, n_hidden=n_regression_layers-1,out_dim=n_regression_weights,activate_all=True
        )
        self.regression_treated = make_sequential(
            in_dim=n_regression_weights, hidden_dim=0, out_dim=1, n_hidden=0
        )
        self.regression_untreated = make_sequential(
            in_dim=n_regression_weights, hidden_dim=0, out_dim=1, n_hidden=0
        )

    def get_riesz_representer(self, data):
        riesz_prediction = self.rr(data.net_input)
        return riesz_prediction

    def _evaluate_regression(self, data):
        t = data.treatments_tensor
        z = self.shared_regression(data.covariates_tensor)
        regression = t * self.regression_treated(z) + (1 - t) * self.regression_untreated(
            z
        )
        return regression

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
        data_treated, data_untreated = data.get_counterfactual_datasets()

        rr_treated = self.get_riesz_representer(data_treated)
        rr_untreated = self.get_riesz_representer(data_untreated)
        rr_output = self.get_riesz_representer(data)
        rr_functional = rr_treated - rr_untreated
        outcome_prediction = self._evaluate_regression(data)

        return rr_output, rr_functional, outcome_prediction
