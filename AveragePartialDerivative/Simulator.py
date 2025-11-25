import numpy as np


def generate_mu_and_sigma(dim=26, mu_first=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mu = np.random.uniform(low=-1.0, high=1.0, size=dim)
    mu[0] = mu_first  # first mean fixed to 1

    # Create a positive semi-definite covariance matrix
    A = np.random.uniform(low=-1.0, high=1.0, size=(dim, dim))
    sigma = A @ A.T  # symmetric PSD covariance matrix

    # Save to files
    np.save("AveragePartialDerivative/parameters/mu.npy", mu)
    np.save("AveragePartialDerivative/parameters/sigma.npy", sigma)

    return mu, sigma


def sample_multivariate_normal(
    n=1000,
    mu_file="AveragePartialDerivative/parameters/mu.npy",
    sigma_file="AveragePartialDerivative/parameters/sigma.npy",
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    # Load parameters
    mu = np.load(mu_file)
    sigma = np.load(sigma_file)

    # Sample N points from MVN(mu, sigma)
    samples = np.random.multivariate_normal(mean=mu, cov=sigma, size=n)

    return samples


if __name__ == "__main__":
    mu, sigma = generate_mu_and_sigma(seed=42)

    print("Mean vector shape:", mu.shape)
    print("Covariance matrix shape:", sigma.shape)
    print("First few mean values:", mu[:5])
