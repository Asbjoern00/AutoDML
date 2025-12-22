from ase_experiment.dataset import Dataset
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
        treatment_coef_file="ate_experiment/LASSO_experiment/propensity_coefficients.npy",
        regression_coef_file="ate_experiment/LASSO_experiment/regression_coefficients.npy",
    ):
        treatment_beta = np.load(treatment_coef_file)
        outcome_beta = np.load(regression_coef_file)

        covariates = np.random.uniform(low=0, high=1, size=(number_of_samples, treatment_beta.shape[0]))
        treatments = cls.treatment_regression(covariates, treatment_beta) + np.random.normal(
            loc=0, scale=1, size=number_of_samples
        )
        noise = np.random.normal(loc=0, scale=1, size=number_of_samples)
        outcomes = cls.outcome_regression(covariates, treatments, outcome_beta) + noise
        data = np.concatenate([outcomes.reshape(-1, 1), treatments.reshape(-1, 1), covariates], axis=1)
        return cls(raw_data=data, outcome_column=0, treatment_column=1)

    @staticmethod
    def outcome_regression(covariates, treatments, beta):
        design_matrix = np.concatenate([treatments.reshape(-1, 1), covariates], axis=1)
        return design_matrix @ beta

    @staticmethod
    def treatment_regression(covariates, beta):
        return covariates @ beta


    def get_counterfactual_datasets(self):
        treated_raw_data = self.raw_data.copy()
        treated_raw_data[:, self.treatment_column] = np.ones_like(treated_raw_data[:, self.treatment_column])
        control_raw_data = self.raw_data.copy()
        control_raw_data[:, self.treatment_column] = np.zeros_like(control_raw_data[:, self.treatment_column])
        return (
            Dataset(treated_raw_data, self.outcome_column, self.treatment_column),
            Dataset(control_raw_data, self.outcome_column, self.treatment_column),
        )
