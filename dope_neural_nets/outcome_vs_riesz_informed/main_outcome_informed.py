import torch
import numpy as np
from dope_neural_nets.model import ModelWrapper
from dope_neural_nets.dataset import Dataset

np.random.seed(42)
torch.manual_seed(42)

truth = 2.121539888279284


## Double training

def run_experiment(data):
    folds = data.split_into_folds(5)
    estimate_components = []
    for j in range(5):
        fit_fold, train_folds = Dataset.get_fit_and_train_folds(folds, j)
        model_wrapper = ModelWrapper()
        model_wrapper.train_outcome_head(train_folds, train_shared_layers=True)
        model_wrapper.train_riesz_head(train_folds, train_shared_layers=False)
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
for i in range(100):
    if i == 8:
        continue
    data = Dataset.load_redrawn_t_replication(i + 1)
    result = run_experiment(data)
    truths.append(data.get_truth())
    results.append(result)
    residuals = [np.abs(result["estimate"] - truth) for result, truth in zip(results, truths)]
    mse = sum(residual**2 for residual in residuals) / len(residuals)
    coverage = sum(result["lower"] <= truth <= result["upper"] for result, truth in zip(results, truths)) / len(results)
    print(
        i, data.get_truth(), "Estimate:", result["estimate"], "RMSE:", mse**0.5, "Coverage:", coverage
    )

import pandas as pd

# estimates = pd.DataFrame(results)
# estimates.to_csv("dope_neural_nets/outcome_vs_riesz_informed/outcome_informed.csv", index=False)
