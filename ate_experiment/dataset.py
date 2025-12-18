import numpy as np


class Dataset:
    def __init__(self, raw_data: np.ndarray, outcome_column: int, treatment_column: int):
        self.raw_data = raw_data
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        covariate_columns = [i for i in range(raw_data.shape[1]) if i not in [outcome_column, treatment_column]]
        self.covariate_columns = covariate_columns

    @classmethod
    def simulate_dataset(cls, size, number_of_covariates):
        covariates = np.random.uniform(low=-1, high=1, size=(size, number_of_covariates))
        propensities = cls.propensity_score(covariates)
        treatments = np.random.binomial(1, propensities, size=size)
        noise = np.random.normal(loc=0, scale=1, size=size)
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
