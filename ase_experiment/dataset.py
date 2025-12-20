import numpy as np
import xgboost as xgb


class Dataset:
    def __init__(self, raw_data: np.ndarray, outcome_column: int, treatment_column: int):
        self.raw_data = raw_data
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        covariate_columns = [i for i in range(raw_data.shape[1]) if i not in [outcome_column, treatment_column]]
        self.covariate_columns = covariate_columns
        self.treatment_shift = 1

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

    @classmethod
    def simulate_dataset(cls, number_of_samples, number_of_covariates):
        covariates = np.random.uniform(low=0, high=2, size=(number_of_samples, number_of_covariates))
        treatments_noise = np.random.normal(loc=0, scale=np.sqrt(2), size=number_of_samples)
        treatments = cls.treatment_regression(covariates) + treatments_noise
        outcomes_noise = np.random.normal(loc=0, scale=1, size=number_of_samples)
        outcomes = cls.outcome_regression(covariates, treatments) + outcomes_noise
        data = np.concatenate([outcomes.reshape(-1, 1), treatments.reshape(-1, 1), covariates], axis=1)
        return cls(raw_data=data, outcome_column=0, treatment_column=1)

    @staticmethod
    def outcome_regression(covariates, treatments):
        X0 = covariates[:, 0]
        return 5 * X0 + 9 * treatments * (X0 + 2) ** 2 + 5 * np.sin(X0 * 3.14) + 25 * treatments

    @staticmethod
    def treatment_regression(covariates):
        X0 = covariates[:, 0]
        return X0**2 - 1

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
    def xgb_dataset(self):
        return xgb.DMatrix(self.raw_data[:, [self.treatment_column] + self.covariate_columns], label=self.outcomes)

    @property
    def xgb_treatment_dataset(self):
        return xgb.DMatrix(self.covariates, label=self.treatments)

    @property
    def xgb_riesz_dataset(self):
        treated_dataset, control_dataset = self.get_counterfactual_datasets()
        data = self.join_datasets([self, treated_dataset, control_dataset])
        labels = [2] * self.raw_data.shape[0] + [1] * self.raw_data.shape[0] + [0] * self.raw_data.shape[0]
        return xgb.DMatrix(data.raw_data[:, [self.treatment_column] + self.covariate_columns], label=np.array(labels))

    def get_counterfactual_datasets(self):
        treated_raw_data = self.raw_data.copy()
        treated_raw_data[:, self.treatment_column] = self.treatments + self.treatment_shift
        control_raw_data = self.raw_data.copy()
        control_raw_data[:, self.treatment_column] = self.treatments
        return (
            Dataset(treated_raw_data, self.outcome_column, self.treatment_column),
            Dataset(control_raw_data, self.outcome_column, self.treatment_column),
        )
