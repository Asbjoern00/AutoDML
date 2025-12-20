from ate_experiment.dataset import Dataset
import numpy as np


class DatasetHighDim(Dataset):
    def __init__(self, raw_data: np.ndarray, outcome_column: int, treatment_column: int):
        super().__init__(
            raw_data=raw_data,
            outcome_column=outcome_column,
            treatment_column=treatment_column,
        )

    @classmethod
    def simulate_dataset(
        cls,
        number_of_samples,
        propensity_coef_file="ate_experiment/LASSO_experiment/propensity_coefficients.npy",
        regression_coef_file="ate_experiment/LASSO_experiment/regression_coefficients.npy",
    ):
        propensity_beta = np.load(propensity_coef_file)
        outcome_beta = np.load(regression_coef_file)


        covariates = np.random.uniform(low=0, high=1, size=(number_of_samples,propensity_beta.shape[0]))
        propensities = cls.propensity_score(covariates, propensity_beta)
        treatments = np.random.binomial(1, propensities, size=number_of_samples)
        noise = np.random.normal(loc=0, scale=1, size=number_of_samples)
        outcomes = cls.outcome_regression(covariates, treatments, outcome_beta) + noise
        data = np.concatenate([outcomes.reshape(-1, 1), treatments.reshape(-1, 1), covariates], axis=1)
        return cls(raw_data=data, outcome_column=0, treatment_column=1)

    @staticmethod
    def outcome_regression(covariates, treatments, beta):
        design_matrix = np.concatenate([treatments.reshape(-1, 1),covariates], axis=1)
        return design_matrix @ beta

    @staticmethod
    def propensity_score(covariates, beta):
        logit = covariates @ beta
        return 1 / (1 + np.exp(-logit))
