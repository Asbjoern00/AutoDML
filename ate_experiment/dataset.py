import numpy as np
import xgboost as xgb
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
        treated_indices = indices[self.treatments == 1]
        control_indices = indices[self.treatments == 0]
        np.random.shuffle(treated_indices)
        np.random.shuffle(control_indices)
        treated_fold_indices = np.array_split(treated_indices, folds)
        control_fold_indices = np.array_split(control_indices, folds)
        folds = [
            self.raw_data[np.concatenate([treated_fold_indices[i], control_fold_indices[i]])] for i in range(folds)
        ]
        return [Dataset(fold, self.outcome_column, self.treatment_column) for fold in folds]

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
            Dataset(train_data, self.outcome_column, self.treatment_column),
            Dataset(test_data, self.outcome_column, self.treatment_column),
        )

    @classmethod
    def join_datasets(cls, datasets):
        raw_data = np.concatenate([dataset.raw_data for dataset in datasets], axis=0)
        return cls(raw_data, datasets[0].outcome_column, datasets[0].treatment_column)

    @classmethod
    def simulate_dataset(cls, number_of_samples, number_of_covariates,seed = None):
        assert number_of_covariates >= 8
        if seed is not None:
            np.random.seed(seed)
        covariates = np.random.uniform(low=0, high=1, size=(number_of_samples, number_of_covariates))
        propensities = cls.propensity_score(covariates)
        treatments = np.random.binomial(1, propensities, size=number_of_samples)
        noise = np.random.normal(loc=0, scale=1, size=number_of_samples)
        outcomes = cls.outcome_regression(covariates, treatments) + noise
        data = np.concatenate([outcomes.reshape(-1, 1), treatments.reshape(-1, 1), covariates], axis=1)
        return cls(raw_data=data, outcome_column=0, treatment_column=1)

    @staticmethod
    def outcome_regression(covariates, treatments):
        X0 = covariates[:, 0]
        X1 = covariates[:, 1]
        X2 = covariates[:, 2]
        X3 = covariates[:, 3]
        X4 = covariates[:, 4]
        X5 = covariates[:, 5]
        treated_regression = X0 + X1**2 + X2 + np.sin(X3 * 3.14) + np.exp(X3 * X4)
        control_regression = X0 + X1**2 + X2**2 + np.cos(X5 * 3.14)
        return treatments * treated_regression + (1 - treatments) * control_regression

    @staticmethod
    def propensity_score(covariates):
        X4 = covariates[:, 4]
        X5 = covariates[:, 5]
        X6 = covariates[:, 6]
        X7 = covariates[:, 7]
        logit = X4 + np.cos(X5*3.14) - np.cos(X6*3.14) + -X4*X7 - 1.5
        return 1 / (1 + np.exp(-logit))

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

    @property
    def xgb_dataset(self):
        return xgb.DMatrix(self.raw_data[:, [self.treatment_column] + self.covariate_columns], label=self.outcomes)

    @property
    def xgb_propensity_dataset(self):
        return xgb.DMatrix(self.covariates, label=self.treatments)

    @property
    def xgb_riesz_dataset(self):
        treated_dataset, control_dataset = self.get_counterfactual_datasets()
        data = self.join_datasets([self, treated_dataset, control_dataset])
        labels = [2] * self.raw_data.shape[0] + [1] * self.raw_data.shape[0] + [0] * self.raw_data.shape[0]
        return xgb.DMatrix(data.raw_data[:, [self.treatment_column] + self.covariate_columns], label=np.array(labels))

    def get_counterfactual_datasets(self):
        treated_raw_data = self.raw_data.copy()
        treated_raw_data[:, self.treatment_column] = np.ones_like(treated_raw_data[:, self.treatment_column])
        control_raw_data = self.raw_data.copy()
        control_raw_data[:, self.treatment_column] = np.zeros_like(control_raw_data[:, self.treatment_column])
        return (
            Dataset(treated_raw_data, self.outcome_column, self.treatment_column),
            Dataset(control_raw_data, self.outcome_column, self.treatment_column),
        )
