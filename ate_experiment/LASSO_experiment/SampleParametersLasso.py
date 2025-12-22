import numpy as np


def sample_parameters_lasso(total_covariates=2000, n_active_regression=20, n_active_propensity=20):

    regression_coefficients = np.concatenate([np.array([1.0]), np.zeros(total_covariates)])
    regression_indices = np.random.choice(np.arange(1, total_covariates + 1), size=n_active_regression, replace=False)
    regression_coefficients[regression_indices] = np.random.uniform(-3, 3, n_active_regression)

    propensity_coefficients = np.zeros(total_covariates)
    propensity_indices = np.random.choice(np.arange(total_covariates), size=n_active_propensity, replace=False)
    propensity_coefficients[propensity_indices] = np.random.uniform(-3, 3, n_active_propensity)

    return regression_coefficients, propensity_coefficients


if __name__ == "__main__":
    regression_coefficients, propensity_coefficients = sample_parameters_lasso()
    np.save("ate_experiment/LASSO_experiment/regression_coefficients.npy", regression_coefficients)
    np.save("ate_experiment/LASSO_experiment/propensity_coefficients.npy", propensity_coefficients)
