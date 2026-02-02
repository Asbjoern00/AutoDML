import numpy as np
from AveragePartialDerivative.Dataset import Dataset
from RieszNet.DOPERieszNetModule import DOPERieszNetModule
from RieszNet.Optimizer import OptimizerParams
from RieszNet.DOPERieszNetDerivative import DOPEDerivativeRieszNetwork
from AveragePartialDerivative.AveragePartialDerivativeFunctional import avg_der_fuctional
import torch
import pandas as pd


def run(n_shared):

    m = 1000
    est_plugin_riesz = np.zeros(m)
    truths = np.zeros(m)

    est_riesz = np.zeros(m)

    var_riesz = np.zeros(m)

    covered_riesz = np.zeros(m)

    upper_ci_riesz = np.zeros(m)

    lower_ci_riesz = np.zeros(m)

    for i in range(m):
        np.random.seed(i)
        torch.manual_seed(i)

        data = pd.read_csv(f"AveragePartialDerivative/BHP_data/redrawn_datasets/data_{i}.csv")

        truths[i] = data["Truth"].iloc[0]

        Y = np.array(data["Y"]).reshape(-1, 1)
        T = np.array(data["T"]).reshape(-1, 1)
        W = np.array(data[data.columns[3:]])

        data = Dataset(np.concatenate([Y, T, W], axis=1), outcome_column=0, treatment_column=1)

        n = data.treatments.shape[0]
        n_features = data.covariates.shape[1]

        network = DOPEDerivativeRieszNetwork(
            avg_der_fuctional,
            features_in=n_features,
            n_shared_layers=n_shared,
            n_regression_layers=2,
            n_riesz_layers=2,
            n_regression_weights=100,
            n_riesz_weights=100,
            hidden_shared=100,
            final_hidden_shared=100,
        )

        optim_regression = OptimizerParams([network.regression, network.shared])
        optim_rr = OptimizerParams([network.rr_head])

        riesz_net = DOPERieszNetModule(network=network, regression_optimizer=optim_regression, rr_optimizer=optim_rr)

        riesz_net.fit(data, informed="regression")

        functional_riesz = riesz_net.get_functional(data).flatten()
        correction_riesz = riesz_net.get_correction(data).flatten()

        est_plugin_riesz[i] = np.mean(functional_riesz)

        est_riesz[i] = np.mean(est_plugin_riesz[i] + correction_riesz)
        var_riesz[i] = np.mean((functional_riesz - est_riesz[i] + correction_riesz) ** 2)
        lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
        upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
        covered_riesz[i] = (lower_ci_riesz[i] < truths[i]) * (truths[i] < upper_ci_riesz[i])

        print(f"plug-in: {est_plugin_riesz[i]}")
        print(f"corrected: {est_riesz[i]}")
        print(f"Outcome informed, n_shared = {n_shared}")
        print(
            f"RMSE : {np.sqrt(np.mean((est_riesz[:i+1]-truths[:i+1])**2))}, coverage = {np.mean(covered_riesz[:i+1])}, bias : {(np.mean((est_riesz[:i+1]-truths[:i+1])))},"
            f"MAE: {np.mean(np.abs(est_riesz[:i+1]-truths[:i+1]))}"
        )
        print(i)

    headers = ["truth", "plugin_estimate_riesz", "riesz_estimate", "riesz_variance", "riesz_lower", "riesz_upper"]

    results = np.array([truths, est_plugin_riesz, est_riesz, var_riesz, lower_ci_riesz, upper_ci_riesz]).T

    np.savetxt(
        f"AveragePartialDerivativeFunctional/results/avg_derivative_dope_{n_shared}.csv",
        results,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )


if __name__ == "__main__":
    run(n_shared=3)
