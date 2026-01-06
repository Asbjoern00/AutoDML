import numpy as np


def sample_parameters_lasso(
    total_covariates=2000,
    n_active_regression=20,
    n_active_propensity=20,
    n_active_intersection=10,
):
    common_covariates = np.random.choice(
        np.arange(total_covariates),
        size=n_active_intersection,
        replace=False,
    )

    remaining_covariates = np.setdiff1d(
        np.arange(total_covariates),
        common_covariates,
        assume_unique=True,
    )

    regression_only = np.random.choice(
        remaining_covariates,
        size=n_active_regression - n_active_intersection,
        replace=False,
    )

    propensity_only = np.random.choice(
        np.setdiff1d(remaining_covariates, regression_only, assume_unique=True),
        size=n_active_propensity - n_active_intersection,
        replace=False,
    )

    regression_covariates = np.concatenate([common_covariates, regression_only])
    propensity_covariates = np.concatenate([common_covariates, propensity_only])

    regression_coefficients = np.zeros(total_covariates + 1)
    regression_coefficients[0] = 1.0  # intercept
    regression_coefficients[regression_covariates + 1] = np.random.uniform(
        -3, 3, size=n_active_regression
    )

    propensity_coefficients = np.zeros(total_covariates)
    propensity_coefficients[propensity_covariates] = np.random.uniform(
        -3, 3, size=n_active_propensity
    )

    return regression_coefficients, propensity_coefficients

if __name__ == "__main__":
    regression_coefficients, propensity_coefficients = sample_parameters_lasso()
    np.save("ate_experiment/DOPE/LASSO_experiment/regression_coefficients.npy", regression_coefficients)
    np.save("ate_experiment/DOPE/LASSO_experiment/propensity_coefficients.npy", propensity_coefficients)
