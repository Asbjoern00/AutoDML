import numpy as np


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
    def simulate_dataset(cls, number_of_samples, number_of_covariates):
        covariates = np.random.uniform(low=-0.5, high=0.5, size=(number_of_samples, number_of_covariates))
        propensities = cls.propensity_score(covariates)
        treatments = np.random.binomial(1, propensities, size=number_of_samples)
        noise = np.random.normal(loc=0, scale=1, size=number_of_samples)
        outcomes = cls.outcome_regression(covariates, treatments) + noise
        data = np.concatenate([outcomes.reshape(-1, 1), treatments.reshape(-1, 1), covariates], axis=1)
        return cls(raw_data=data, outcome_column=0, treatment_column=1)

    @staticmethod
    def outcome_regression(covariates, treatments):
        treated_regression = covariates[:, 0] ** 2 + covariates[:, 2] + covariates[:, 3]
        control_regression = covariates[:, 0] ** 2 + covariates[:, 2] + covariates[:, 3]
        return treatments * treated_regression + (1 - treatments) * control_regression

    @staticmethod
    def propensity_score(covariates):
        logit = covariates[:, 0] ** 2 + covariates[:, 2] + covariates[:, 3]
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
