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

    @classmethod
    def from_csv(cls, path):
        data = np.loadtxt(path)
        return cls(
            treatments=data[:, 0].reshape(-1, 1).astype(np.float32),
            outcomes=data[:, 1].reshape(-1, 1).astype(np.float32),
            counterfactual_outcomes=data[:, 2].reshape(-1, 1).astype(np.float32),
            noiseless_untreated_outcomes=data[:, 3].reshape(-1, 1).astype(np.float32),
            noiseless_treated_outcomes=data[:, 4].reshape(-1, 1).astype(np.float32),
            covariates=data[:, 5:].astype(np.float32),
        )

    @classmethod
    def load_chernozhukov_replication(cls, index):
        path = "average_treatment_effect/data/chernozhukov_ihdp_data/ihdp_" + str(index) + ".csv"
        return cls.from_csv(path)

    def get_average_treatment_effect(self):
        return np.mean(self.noiseless_treated_outcomes - self.noiseless_untreated_outcomes)