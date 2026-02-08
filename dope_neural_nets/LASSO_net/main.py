import torch
import numpy as np
from dope_neural_nets.LASSO_net.model import ModelWrapper
from dope_neural_nets.dataset import Dataset

np.random.seed(42)
torch.manual_seed(42)

penalties = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

def run_experiment(data):
    estimate_components = []
    best = 1e6
    for penalty in penalties:
        model_wrapper_ = ModelWrapper(in_=25, hidden_size=100, n_shared=3, n_not_shared=2)
        res = model_wrapper_.train_outcome_head(data, train_shared_layers=True, lr=1e-3, l1_penalty=penalty)
        if res < best:
            best = res
            model_wrapper = model_wrapper_
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
        "MAE",
        mae,
        "Coverage:",
        coverage,
    )


