from dataclasses import dataclass
import numpy as np
from typing import Optional, List
import torch
from AveragePartialDerivative.Simulator import sample_multivariate_normal


@dataclass
class Dataset:
    treatments: np.ndarray
    outcomes: np.ndarray
    covariates: np.ndarray
    folds: Optional[List[np.ndarray]] = None
    avg_partial_derivative : Optional[np.float64] = None

    @classmethod
    def from_sample(cls, n=1000):
        predictors, outcomes, avg_partial_derivative = sample_multivariate_normal(n=n)
        return cls(
            treatments=predictors[:, 0].reshape(-1, 1).astype(np.float32),
            outcomes=outcomes.reshape(-1, 1).astype(np.float32),
            covariates=predictors[:, 1:].astype(np.float32),
            avg_partial_derivative=avg_partial_derivative,
        )

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
            covariates=self.covariates[selected_indices]
        )
