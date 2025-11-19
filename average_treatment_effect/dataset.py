from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class Dataset:
    treatments: Optional[np.ndarray] = None
    outcomes: Optional[np.ndarray] = None
    counterfactual_outcomes: Optional[np.ndarray] = None
    noiseless_untreated_outcomes: Optional[np.ndarray] = None
    noiseless_treated_outcomes: Optional[np.ndarray] = None
    covariates: Optional[np.ndarray] = None

    def get_average_treatment_effect(self):
        return np.mean(
            self.noiseless_treated_outcomes - self.noiseless_untreated_outcomes
        )
