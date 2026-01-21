import torch
import numpy as np
from dope_neural_nets.outcome_informed_ihdp.model import ModelWrapper
from dope_neural_nets.dataset import Dataset

np.random.seed(42)
torch.manual_seed(42)

truth = 2.121539888279284


def run_experiment(data):
    folds = data.split_into_folds(5)
    estimate_components = []
    for j in range(5):
        fit_fold, train_folds = Dataset.get_fit_and_train_folds(folds, j)
        model_wrapper = ModelWrapper(in_=10, hidden_size=100, n_shared=2, n_not_shared=1)
        model_wrapper.train_as_riesz_net(train_folds, tmle_w=0, rr_w=10, mse_w=1)
        estimate_components.append(model_wrapper.get_estimate_components(fit_fold))
    estimate_components = torch.concat(estimate_components, dim=0)
    estimate = torch.mean(estimate_components).item()
    variance = torch.var(estimate_components).item()
    return {
        "estimate": estimate,
        "variance": variance,
        "lower": estimate - 1.96 * (variance / data.raw_data.shape[0]) ** 0.5,
        "upper": estimate + 1.96 * (variance / data.raw_data.shape[0]) ** 0.5,
    }


results = []
truths = []
for i in range(1000):
    data = Dataset.simulate_dataset(1000, 10)
    result = run_experiment(data)
    truths.append(truth)
    results.append(result)
    residuals = [np.abs(result["estimate"] - truth) for result, truth in zip(results, truths)]
    mse = sum(residual**2 for residual in residuals) / len(residuals)
    coverage = sum(result["lower"] <= truth <= result["upper"] for result, truth in zip(results, truths)) / len(results)
    print(i, truth, "Estimate:", result["estimate"], "RMSE:", mse**0.5, "Coverage:", coverage)

import pandas as pd

estimates = pd.DataFrame(results)
estimates.to_csv("dope_neural_nets/rr_weight/rrw_10.csv", index=False)
