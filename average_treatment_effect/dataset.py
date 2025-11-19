from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    treatments: np.ndarray
    outcomes: np.ndarray
    counterfactual_outcomes: np.ndarray
    noiseless_untreated_outcomes: np.ndarray
    noiseless_treated_outcomes: np.ndarray
    covariates: np.ndarray

    def get_average_treatment_effect(self):
        return np.mean(
            self.noiseless_treated_outcomes - self.noiseless_untreated_outcomes
        )
