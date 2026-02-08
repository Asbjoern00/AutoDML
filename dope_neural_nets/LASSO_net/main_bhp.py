import torch
import numpy as np
from dope_neural_nets.LASSO_net.model_bhp import ModelWrapper
from dope_neural_nets.LASSO_net.dataset_bhp import Dataset

np.random.seed(42)
torch.manual_seed(42)

penalties = [0, 1e-3, 1e-2, 1e-1, 1]

def run_experiment(data):
    estimate_components = []
    train, test = data.test_train_split(0.8)
    penalty = 0
    best = 1e6
    for pen in penalties:
        model_wrapper_ = ModelWrapper(in_=50, hidden_size=100, n_shared=3, n_not_shared=2)
        model_wrapper_.train_outcome_head(train, train_shared_layers=True, lr=1e-3, l1_penalty=pen)
        res = model_wrapper_._get_mse_loss(test.net_input, test.outcomes_tensor).item()
        print(pen, res)
        if res < best:
            best = res
            penalty = pen
    model_wrapper = ModelWrapper(in_=50, hidden_size=100, n_shared=3, n_not_shared=2)
    model_wrapper.train_outcome_head(data, train_shared_layers=True, lr=1e-3, l1_penalty=penalty)
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
    mae = sum(abs(residual) for residual in residuals) / len(residuals)
    bias = sum(residuals) / len(residuals)
    mean_est = sum(result["estimate"] for result in results) / len(results)
    variance = sum((result["estimate"] - mean_est) ** 2 for result in results) / len(results)
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
        "MAE",
        mae,
        "Coverage:",
        coverage,
    )


import pandas as pd

estimates = pd.DataFrame(results)
estimates.to_csv("dope_neural_nets/LASSO_net/lasso_net.csv", index=False)
