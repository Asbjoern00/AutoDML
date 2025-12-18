from dataclasses import dataclass
import xgboost as xgb
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
        path = "ihdp_average_treatment_effect/data/chernozhukov_ihdp_data/ihdp_" + str(index) + ".csv"
        return cls.from_csv(path)

    def get_average_treatment_effect(self):
        return np.mean(self.noiseless_treated_outcomes - self.noiseless_untreated_outcomes)

    def get_as_tensor(self, attributes):
        return torch.from_numpy(getattr(self, attributes))

    def split_into_folds(self, folds):
        n_samples = self.treatments.shape[0]
        indices = np.arange(n_samples, dtype=int)
        treated_indices = indices[self.treatments[:, 0] == 1]
        control_indices = indices[self.treatments[:, 0] == 0]
        np.random.shuffle(treated_indices)
        np.random.shuffle(control_indices)
        treated_folds = np.array_split(treated_indices, folds)
        control_folds = np.array_split(control_indices, folds)
        self.folds = [np.concatenate([treated_folds[i], control_folds[i]]) for i in range(folds)]

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

    def to_xgb_dataset(self):
        treatments = self.treatments[:, 0]
        outcome_dataset_0 = xgb.DMatrix(self.covariates[treatments == 0, :], label=self.outcomes[treatments == 0, 0])
        outcome_dataset_1 = xgb.DMatrix(self.covariates[treatments == 1, :], label=self.outcomes[treatments == 1, 0])
        treatment_dataset = xgb.DMatrix(self.covariates, label=self.treatments)
        full_covariates = xgb.DMatrix(self.covariates)
        return {
            "outcome_dataset_0": outcome_dataset_0,
            "outcome_dataset_1": outcome_dataset_1,
            "treatment_dataset": treatment_dataset,
            "full_covariates": full_covariates,
        }

    def to_riesz_xgb_dataset(self):
        treatments = np.concatenate(
            [self.treatments, np.zeros_like(self.treatments), np.ones_like(self.treatments)], axis=0
        )
        covariates = np.concatenate([self.covariates] * 3, axis=0)
        label = np.concatenate(
            [2 + np.zeros_like(self.treatments), np.zeros_like(self.treatments), 1 + np.zeros_like(self.treatments)],
            axis=0,
        )
        return {
            "training_data": xgb.DMatrix(np.concatenate([treatments, covariates], axis=1), label=label),
            "test_data": xgb.DMatrix(np.concatenate([self.treatments, self.covariates], axis=1)),
        }
