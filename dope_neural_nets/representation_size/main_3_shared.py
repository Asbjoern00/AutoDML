import torch
import numpy as np
from dope_neural_nets.representation_size.model import ModelWrapper
from dope_neural_nets.dataset import Dataset

np.random.seed(42)
torch.manual_seed(42)


def run_experiment(data):
    estimate_components = []
    model_wrapper = ModelWrapper(in_=25, hidden_size=100, n_shared=3, n_not_shared=2)
    model_wrapper.train_outcome_head(data, train_shared_layers=True, lr=1e-3)
    model_wrapper.train_riesz_head(data, train_shared_layers=False, lr=1e-3)
    estimate_components.append(model_wrapper.get_estimate_components(data))
    estimate_components = torch.concat(estimate_components, dim=0)
    estimate = torch.mean(estimate_components).item()
    variance = torch.var(estimate_components).item()
    return {
        "truth": data.get_truth(),
        "estimate": estimate,
        "variance": variance,
        "lower": estimate - 1.96 * (variance / data.raw_data.shape[0]) ** 0.5,
        "upper": estimate + 1.96 * (variance / data.raw_data.shape[0]) ** 0.5,
    }


results = []
for i in range(1000):
    data = Dataset.load_chernozhukov_replication(i + 1)
    result = run_experiment(data)
    results.append(result)
    residuals = [result["estimate"] - result["truth"] for result in results]
    MSE = sum(residual**2 for residual in residuals) / len(residuals)
    bias = sum(residuals) / len(residuals)
    mean_est = sum(result["estimate"] for result in results) / len(results)
    variance = sum((result["estimate"] - mean_est)**2 for result in results) / len(results)
    coverage = sum(result["lower"] <= result["truth"] <= result["upper"] for result in results) / len(results)
    print(
        i,
        data.get_truth(),
        "Estimate:",
        result["estimate"],
        "RMSE:",
        MSE**0.5,
        "Bias:",
        bias,
        "Variance",
        variance,
        "Coverage:",
        coverage,
    )


import pandas as pd

estimates = pd.DataFrame(results)
estimates.to_csv("dope_neural_nets/representation_size/3_shared.csv", index=False)
