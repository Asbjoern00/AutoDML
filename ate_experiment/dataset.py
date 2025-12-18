import numpy as np


class Dataset:
    def __init__(self, raw_data: np.ndarray, outcome_column: int, treatment_column: int):
        self.raw_data = raw_data
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        covariate_columns = [i for i in range(raw_data.shape[1]) if i not in [outcome_column, treatment_column]]
        self.covariate_columns = covariate_columns

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
    def outcome(self):
        return self.raw_data[:, self.outcome_column]

    @property
    def treatment(self):
        return self.raw_data[:, self.treatment_column]

    @property
    def covariates(self):
        return self.raw_data[:, self.covariate_columns]
