import numpy as np
import xgboost as xgb
import torch


class Dataset:
    def __init__(
        self, raw_data: np.ndarray, outcome_column: int, treatment_column: int, covariate_columns: list = None
    ):
        self.raw_data = raw_data
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        if not covariate_columns:
            covariate_columns = [i for i in range(raw_data.shape[1]) if i not in [outcome_column, treatment_column]]
        self.covariate_columns = covariate_columns

    @classmethod
    def from_csv(cls, path):
        data = np.loadtxt(path, skiprows=1, delimiter=",")
        data = data[:, 1:]
        return cls(
            raw_data=data,
            treatment_column=1,
            outcome_column=0,
        )

    def get_truth(self):
        return -0.6

    @classmethod
    def load_chernozhukov_replication(cls, index):
        path = "AveragePartialDerivative/BHP_data/redrawn_datasets/data_" + str(index) + ".csv"
        return cls.from_csv(path)

    @classmethod
    def load_redrawn_t_replication(cls, index):
        path = "ihdp_average_treatment_effect/data/redrawn_t/ihdp_" + str(index) + ".csv"
        return cls.from_csv(path)

    @classmethod
    def get_fit_and_train_folds(cls, folds, fit_index):
        train_data = cls.join_datasets([folds[i] for i in range(len(folds)) if i != fit_index])
        return folds[fit_index], train_data

    def split_into_folds(self, folds):
        number_of_samples = self.raw_data.shape[0]
        indices = np.arange(number_of_samples, dtype=int)
        treated_indices = indices[self.treatments == 1]
        control_indices = indices[self.treatments == 0]
        np.random.shuffle(treated_indices)
        np.random.shuffle(control_indices)
        treated_fold_indices = np.array_split(treated_indices, folds)
        control_fold_indices = np.array_split(control_indices, folds)
        folds = [
            self.raw_data[np.concatenate([treated_fold_indices[i], control_fold_indices[i]])] for i in range(folds)
        ]
        return [Dataset(fold, self.outcome_column, self.treatment_column, self.covariate_columns) for fold in folds]

    def test_train_split(self, train_proportion):
        number_of_samples = self.raw_data.shape[0]
        indices = np.arange(number_of_samples, dtype=int)
        treated_indices = indices[self.treatments == 1]
        control_indices = indices[self.treatments == 0]

        np.random.shuffle(treated_indices)
        np.random.shuffle(control_indices)

        treated_train_indices = treated_indices[: int(train_proportion * len(treated_indices))]
        control_train_indices = control_indices[: int(train_proportion * len(control_indices))]
        train_data = self.raw_data[np.concatenate([treated_train_indices, control_train_indices]), :]

        treated_test_indices = treated_indices[int(train_proportion * len(treated_indices)) :]
        control_test_indices = control_indices[int(train_proportion * len(control_indices)) :]
        test_data = self.raw_data[np.concatenate([treated_test_indices, control_test_indices]), :]

        return (
            Dataset(train_data, self.outcome_column, self.treatment_column, self.covariate_columns),
            Dataset(test_data, self.outcome_column, self.treatment_column, self.covariate_columns),
        )

    @classmethod
    def join_datasets(cls, datasets):
        raw_data = np.concatenate([dataset.raw_data for dataset in datasets], axis=0)
        return cls(raw_data, datasets[0].outcome_column, datasets[0].treatment_column, datasets[0].covariate_columns)

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
        return torch.from_numpy(outcomes).reshape(-1, 1)

    @property
    def treatments_tensor(self):
        treatments = self.treatments.astype(np.float32)
        return torch.from_numpy(treatments).reshape(-1, 1)

    @property
    def covariates_tensor(self):
        covariates = self.covariates.astype(np.float32)
        return torch.from_numpy(covariates)

    @property
    def net_input(self):
        return torch.from_numpy(self.raw_data[:, [self.treatment_column] + self.covariate_columns].astype(np.float32))
