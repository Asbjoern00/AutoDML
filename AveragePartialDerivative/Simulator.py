import numpy as np


def generate_beta_and_sigma(dim=26, beta_first=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    beta = np.random.uniform(low=-1.0, high=1.0, size=dim)
    beta[0] = beta_first  # first mean fixed to 1

    # Create a positive semi-definite covariance matrix
    A = np.random.uniform(low=-1.0, high=1.0, size=(dim, dim))
    sigma = A @ A.T  # symmetric PSD covariance matrix

    # Save to files
    np.save("AveragePartialDerivative/parameters/beta.npy", beta)
    np.save("AveragePartialDerivative/parameters/sigma.npy", sigma)

    return beta, sigma


def sample_multivariate_normal(
    n=1000,
    beta_file="AveragePartialDerivative/parameters/beta.npy",
    sigma_file="AveragePartialDerivative/parameters/sigma.npy",
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    # Load parameters
    beta = np.load(beta_file)
    sigma = np.load(sigma_file)

    # Sample covariates from MVN(sigma)
    predictors = np.random.multivariate_normal(mean=np.zeros(beta.shape[0]), cov=sigma, size=n)
    outcomes = np.random.normal(loc=predictors @ beta, scale=1, size=n)

    return predictors, outcomes, beta[0]


if __name__ == "__main__":
    mu, sigma = generate_beta_and_sigma(seed=42)

    print("Mean vector shape:", mu.shape)
    print("Covariance matrix shape:", sigma.shape)
    print("First few mean values:", mu[:5])
    samples = sample_multivariate_normal()
