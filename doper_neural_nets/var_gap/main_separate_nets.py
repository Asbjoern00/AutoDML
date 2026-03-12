import torch
import numpy as np
from doper_neural_nets.var_gap.model import ModelWrapper
from doper_neural_nets.var_gap.dataset import Dataset
import pandas as pd

n = 2000
p = 2


def run_experiment(data, beta):
    fit, est = data.test_train_split(0.5)
    model_wrapper = ModelWrapper(in_=p, hidden_size=100, n_shared=3, n_not_shared=2, type_="separate_nets")
    model_wrapper.train_outcome_head(fit, lr=1e-3, train_shared_layers=True, epochs=1000, wd=1e-2, batch_size=100)
    model_wrapper.train_riesz_head(fit, lr=1e-3, train_shared_layers=True, epochs=1000, wd=1e-2, batch_size=100)
    estimate_components = model_wrapper.get_estimate_components(est)
    estimate = torch.mean(estimate_components).item()
    return {
        "truth": data.get_truth(),
        "estimate": estimate,
        "beta": beta,
    }


def run(beta, iterations):
    results = []
    output_file = f"doper_neural_nets/var_gap/separate_nets{beta}.csv"
    for i in iterations:
        np.random.seed(i)
        torch.manual_seed(i)
        data = Dataset.simulate_dataset(n, p, beta)
        result = run_experiment(data, beta)
        results.append(result)
        residuals = [np.abs(result["estimate"] - result["truth"]) for result in results]
        MSE = sum(residual**2 for residual in residuals) / len(residuals)
        print(i, beta, "nMSE", MSE * n / 2)

        df = pd.DataFrame([result])
        df.to_csv(output_file, mode="a", header=not pd.io.common.file_exists(output_file), index=False)


if __name__ == "__main__":
    beta = [0.05, 0.5, 0.1, 0.03, 0.3]
    iterations = [[j + (100 * i) for j in range(100)] for i in range(10)]
    for i in iterations:
        for b in beta:
            run(b, i)
