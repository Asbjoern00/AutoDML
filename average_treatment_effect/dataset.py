from dataclasses import dataclass
import numpy as np
from typing import Optional, List
import torch


@dataclass
class Dataset:
    treatments: np.ndarray
    outcomes: np.ndarray
    counterfactual_outcomes: np.ndarray
    noiseless_untreated_outcomes: np.ndarray
    noiseless_treated_outcomes: np.ndarray
    covariates: np.ndarray
    folds: Optional[List[np.ndarray]] = None

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

    def get_as_tensor(self, attributes):
        return torch.from_numpy(getattr(self, attributes))

    def split_into_folds(self, folds):
        n_samples = self.treatments.shape[0]
        indices = np.arange(n_samples, dtype=int)
        np.random.shuffle(indices)
        self.folds = np.array_split(indices, folds)

    def get_folds(self, folds):
        if self.folds is None:
            raise ValueError("You must first split the dataset into folds by calling .split_into_folds()")
        selected_indices = np.concatenate([self.folds[i - 1] for i in folds]).astype(int)
        return Dataset(
            treatments=self.treatments[selected_indices],
            outcomes=self.outcomes[selected_indices],
            counterfactual_outcomes=self.counterfactual_outcomes[selected_indices],
            noiseless_untreated_outcomes=self.noiseless_untreated_outcomes[selected_indices],
            noiseless_treated_outcomes=self.noiseless_treated_outcomes[selected_indices],
            covariates=self.covariates[selected_indices],
        )
