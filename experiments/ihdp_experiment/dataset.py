import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class IHDPDataset:
    treatments: np.ndarray
    outcomes: np.ndarray
    counterfactual_outcomes: np.ndarray
    noiseless_untreated_outcomes: np.ndarray
    noiseless_treated_outcomes: np.ndarray
    covariates: np.ndarray

    @classmethod
    def load_chernozhukov_replication(cls, index):
        data = np.loadtxt("data/chernozhukov_ihdp_data/ihdp_" + str(index) + ".csv")
        return IHDPDataset(
            treatments=data[:, 0].reshape(-1, 1).astype(np.float32),
            outcomes=data[:, 1].reshape(-1, 1).astype(np.float32),
            counterfactual_outcomes=data[:, 2].reshape(-1, 1).astype(np.float32),
            noiseless_untreated_outcomes=data[:, 3].reshape(-1, 1).astype(np.float32),
            noiseless_treated_outcomes=data[:, 4].reshape(-1, 1).astype(np.float32),
            covariates=data[:, 5:].astype(np.float32),
        )

    def get_average_treatment_effect(self):
        return np.mean(
            self.noiseless_treated_outcomes - self.noiseless_untreated_outcomes
        )

    def get_as_tensor(self, attribute):
        return torch.from_numpy(getattr(self, attribute)).float()
