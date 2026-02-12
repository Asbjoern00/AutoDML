import torch
import numpy as np
from dope_neural_nets.LASSO_net.better_than_xgboost.model_ase import ModelWrapper
from dope_neural_nets.LASSO_net.better_than_xgboost.dataset_ase import Dataset
import pandas as pd
import multiprocessing as mp


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _run_iteration(data):
    estimate_components = []
    model_wrapper = ModelWrapper(in_=1, hidden_size=100, n_shared=3, n_not_shared=2)
    model_wrapper.train_outcome_head(data, train_shared_layers=True, lr=1e-3, l1_penalty=100, wd=1e-3)
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


def run_experiment(indices):
    results = []
    for i in indices:
        np.random.seed(i)
        torch.manual_seed(i)
        data = Dataset.simulate_dataset(1000, 1)
        result = _run_iteration(data)
        results.append(result)
        _print_diagnostics(results)
        return results


def _print_diagnostics(results):
    residuals = [result["estimate"] - result["truth"] for result in results]
    MSE = sum(residual**2 for residual in residuals) / len(residuals)
    mae = sum(abs(residual) for residual in residuals) / len(residuals)
    bias = sum(residuals) / len(residuals)
    mean_est = sum(result["estimate"] for result in results) / len(results)
    coverage = sum(result["lower"] <= result["truth"] <= result["upper"] for result in results) / len(results)
    print(
        "RMSE:",
        MSE**0.5,
        "Bias:",
        bias,
        "MAE",
        mae,
        "Coverage:",
        coverage,
    )


if __name__ == "__main__":
    indices = [i for i in range(1000)]
    chunk_size = 1
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
    mp.set_start_method("spawn", force=True)
    n_proc = 4
    result = []
    for i in range(250):
        with mp.Pool(processes=n_proc) as p:
            results = p.map(run_experiment, chunks[i * 4 : (i + 1) * 4])
        for res in results:
            result += res
        print("Total Iterations", len(result))
        _print_diagnostics(result)
    df = pd.DataFrame(result)
    df.to_csv("dope_neural_nets/LASSO_net/lasso_net_ase.csv", index=False)
