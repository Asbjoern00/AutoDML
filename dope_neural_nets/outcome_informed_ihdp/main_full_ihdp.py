import torch
import numpy as np
from dope_neural_nets.model import ModelWrapper
from dope_neural_nets.dataset import Dataset

np.random.seed(42)
torch.manual_seed(42)


def run_experiment(data):
    folds = data.split_into_folds(5)
    estimate_components = []
    for j in range(5):
        fit_fold, train_folds = Dataset.get_fit_and_train_folds(folds, j)
        model_wrapper = ModelWrapper(in_=25, hidden_size=100, n_shared=3, n_not_shared=2)
        model_wrapper.train_outcome_head(train_folds, train_shared_layers=True, lr=1e-3)
        model_wrapper.train_riesz_head(train_folds, train_shared_layers=False, lr=1e-3)
        estimate_components.append(model_wrapper.get_estimate_components(fit_fold))
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
    residuals = [np.abs(result["estimate"] - result["truth"]) for result in results]
    MAE = sum(residual for residual in residuals) / len(residuals)
    coverage = sum(result["lower"] <= result["truth"] <= result["upper"] for result in results) / len(results)
    print(i, data.get_truth(), "Estimate:", result["estimate"], "MAE:", MAE, "Coverage:", coverage)


import pandas as pd

estimates = pd.DataFrame(results)
estimates.to_csv("dope_neural_nets/outcome_informed_ihdp/outcome_informed_ihdp_full.csv", index=False)
