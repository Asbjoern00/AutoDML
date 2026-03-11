import torch
import numpy as np
from doper_neural_nets.var_gap.model import ModelWrapper
from doper_neural_nets.var_gap.dataset import Dataset

n = 10000
p = 2
beta = 2

import pandas as pd

output_file = f"doper_neural_nets/var_gap/separate_nets_{beta}.csv"


def run_experiment(data):
    fit, est = data.test_train_split(0.5)
    model_wrapper = ModelWrapper(in_=p, hidden_size=100, n_shared=3, n_not_shared=2, type_="separate_nets")
    model_wrapper.train_outcome_head(fit, lr=1e-3, train_shared_layers=True, epochs=1000, wd=1e-3, batch_size=64)
    model_wrapper.train_riesz_head(fit, lr=1e-3, train_shared_layers=True, epochs=1000, wd=1e-3, batch_size=1000)
    estimate_components = model_wrapper.get_estimate_components(est)
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
    data = Dataset.simulate_dataset(n, p)
    result = run_experiment(data)
    results.append(result)
    residuals = [np.abs(result["estimate"] - result["truth"]) for result in results]
    MAE = sum(residual for residual in residuals) / len(residuals)
    MSE = sum(residual**2 for residual in residuals) / len(residuals)
    coverage = sum(result["lower"] <= result["truth"] <= result["upper"] for result in results) / len(results)
    print(i, "nMSE", MSE * n / 2)

    df = pd.DataFrame([result])
    df.to_csv(output_file, mode="a", header=not pd.io.common.file_exists(output_file), index=False)
