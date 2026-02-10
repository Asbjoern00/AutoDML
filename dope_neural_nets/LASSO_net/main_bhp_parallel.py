import torch
import numpy as np
from dope_neural_nets.LASSO_net.model_bhp import ModelWrapper
from dope_neural_nets.LASSO_net.dataset_bhp import Dataset
import pandas as pd
import multiprocessing as mp


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _run_iteration(data):
    estimate_components = []
    #train, test = data.test_train_split(0.8)
    penalty = 0
    best = 1e6
    #for pen in [0, 1e-3, 1e-2, 1e-1, 1]:
    #    model_wrapper_ = ModelWrapper(in_=50, hidden_size=100, n_shared=3, n_not_shared=2)
    #    model_wrapper_.train_outcome_head(train, train_shared_layers=True, lr=1e-3, l1_penalty=pen, wd=1e-3)
    #    res = model_wrapper_._get_mse_loss(test.net_input, test.outcomes_tensor).item()
    #    print(pen, res)
    #    if res < best:
    #        best = res
    #        penalty = pen
    model_wrapper = ModelWrapper(in_=50, hidden_size=100, n_shared=3, n_not_shared=2)
    model_wrapper.train_outcome_head(data, train_shared_layers=True, lr=1e-3, l1_penalty=0, wd=1e-3)
    model_wrapper.train_riesz_head(data, train_shared_layers=False, lr=1e-3, wd=1e-3)
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
        data = Dataset.load_chernozhukov_replication(i)
        result = _run_iteration(data)
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
    return results


if __name__ == "__main__":
    indices = [i for i in range(1000)]
    chunk_size = 143
    chunks = [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]
    mp.set_start_method("spawn", force=True)
    n_proc = 7

    with mp.Pool(processes=n_proc) as p:
        results = p.map(run_experiment, chunks)
    result = []
    for res in results:
        result += res
    df = pd.DataFrame(result)
    df.to_csv("dope_neural_nets/LASSO_net/lasso_net_bhp_no_lasso.csv", index=False)
