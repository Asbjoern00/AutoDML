import torch
import numpy as np
from doper_neural_nets.var_gap.model import ModelWrapper
from doper_neural_nets.var_gap.dataset import Dataset

n = 300
p = 2


def run_experiment(data):
    model_wrapper = ModelWrapper(in_=p, hidden_size=100, n_shared=1, n_not_shared=2)
    model_wrapper.train_outcome_head(data, lr=1e-3, train_shared_layers=True, epochs=250, wd=0.01, l1=0.1)
    model_wrapper.train_riesz_head(data, lr=1e-3, train_shared_layers=False, epochs=250, wd=0.01)
    estimate_components = model_wrapper.get_estimate_components(data)
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
    np.random.seed(i)
    torch.manual_seed(i)
    data = Dataset.simulate_dataset(n,p)
    result = run_experiment(data)
    results.append(result)
    residuals = [np.abs(result["estimate"] - result["truth"]) for result in results]
    MAE = sum(residual for residual in residuals) / len(residuals)
    MSE = sum(residual**2 for residual in residuals) / len(residuals)
    coverage = sum(result["lower"] <= result["truth"] <= result["upper"] for result in results) / len(results)
    print(i, "nMSE", MSE*n)


import pandas as pd

estimates = pd.DataFrame(results)
estimates.to_csv("doper_neural_nets/var_gap/outcome_informed.csv", index=False)
