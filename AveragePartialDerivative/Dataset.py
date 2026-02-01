import numpy as np
import torch

class Dataset:
    def __init__(self, raw_data: np.ndarray, outcome_column: int, treatment_column: int):
        self.raw_data = raw_data
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        covariate_columns = [i for i in range(raw_data.shape[1]) if i not in [outcome_column, treatment_column]]
        self.covariate_columns = covariate_columns

    @classmethod
    def get_fit_and_train_folds(cls, folds, fit_index):
        train_data = cls.join_datasets([folds[i] for i in range(len(folds)) if i != fit_index])
        return folds[fit_index], train_data

    def split_into_folds(self, folds):
        number_of_samples = self.raw_data.shape[0]
        indices = np.arange(number_of_samples, dtype=int)
        np.random.shuffle(indices)
        indices = np.array_split(indices, folds)
        folds = [self.raw_data[indices[i]] for i in range(folds)]
        return [Dataset(fold, self.outcome_column, self.treatment_column) for fold in folds]

    def test_train_split(self, train_proportion):
        number_of_samples = self.raw_data.shape[0]
        indices = np.arange(number_of_samples, dtype=int)
        np.random.shuffle(indices)
        train_indices = indices[: int(train_proportion * len(indices))]
        train_data = self.raw_data[train_indices]
        test_indices = indices[int(train_proportion * len(indices)) :]
        test_data = self.raw_data[test_indices]

        return (
            Dataset(train_data, self.outcome_column, self.treatment_column),
            Dataset(test_data, self.outcome_column, self.treatment_column),
        )

    @classmethod
    def join_datasets(cls, datasets):
        raw_data = np.concatenate([dataset.raw_data for dataset in datasets], axis=0)
        return cls(raw_data, datasets[0].outcome_column, datasets[0].treatment_column)

    @property
    def outcomes(self):
        return self.raw_data[:, self.outcome_column]

    @property
    def treatments(self):
        return self.raw_data[:, self.treatment_column]

    @property
    def covariates(self):
        return self.raw_data[:, self.covariate_columns]

    @property
    def outcomes_tensor(self):
        outcomes = self.outcomes.astype(np.float32)
        return torch.from_numpy(outcomes).reshape(-1,1)

    @property
    def treatments_tensor(self):
        treatments = self.treatments.astype(np.float32)
        return torch.from_numpy(treatments).reshape(-1,1)

    @property
    def covariates_tensor(self):
        covariates = self.covariates.astype(np.float32)
        return torch.from_numpy(covariates)

    def get_counterfactual_datasets(self, epsilon = 0.001):
        raw_data_down = self.raw_data.copy()
        raw_data_down[:, self.treatment_column] = self.treatments - epsilon
        raw_data_up = self.raw_data.copy()
        raw_data_up[:, self.treatment_column] = self.treatments + epsilon
        return (
            Dataset(raw_data_down, self.outcome_column, self.treatment_column),
            Dataset(raw_data_up, self.outcome_column, self.treatment_column),
        )


