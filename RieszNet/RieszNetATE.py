import torch
import torch.nn as nn
import torch.nn.functional as F
from RieszNet.utilities import make_sequential

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

    def _forward_shared(self, x):
        return self.shared(x)

    def _evaluate_riesz(self, x):
        z = self._forward_shared(x)
        return self.rr_head(z)

    def forward(self, data):
        # Riesz functional
        rr_functional = self.functional(data, self._evaluate_riesz)

        # Shared representation
        z = self._forward_shared(data)

        # Heads
        rr_output = self.rr_head(z)
        y_treated = self.treated_head(z)
        y_untreated = self.untreated_head(z)

        # Treatment indicator assumed in column 0
        t = data[:, [0]]
        outcome_prediction = t * y_treated + (1 - t) * y_untreated

        return rr_output, rr_functional, outcome_prediction, self.epsilon
