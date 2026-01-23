import numpy as np
import torch
from dope_neural_nets.dataset import Dataset
from RieszNet.RieszNetModule import RieszNetModule
from RieszNet.Optimizer import Optimizer
from RieszNet.RieszNetATE import ATERieszNetwork
from RieszNet.Loss import RieszNetLoss
from average_treatment_effect.Functional.ATEFunctional import ate_functional
from ihdp_average_treatment_effect.dataset import Dataset as CData
import os
import pandas as pd

def run(n_riesz):

    m = 1000
    rr_weights = 2.0**np.arange(-5, 0, step=1)
    tmle_weight = 0.1


    for rr_weight in rr_weights:

        if os.path.exists(f"ate_experiment/neural_net_experiment/results/wTMLE/rr_w_{rr_weight}_TMLE_0.1.csv"):
            res = pd.read_csv(f"ate_experiment/neural_net_experiment/results/wTMLE/rr_w_{rr_weight}_TMLE_0.1.csv")
            est_plugin_riesz = np.array(res["plugin_estimate_riesz"])
            est_riesz = np.array(res["riesz_estimate"])
            var_riesz = np.array(res["riesz_variance"])
            upper_ci_riesz = np.array(res["riesz_upper"])
            lower_ci_riesz = np.array(res["riesz_lower"])
            truths = np.array(res["truth"])
            covered_riesz = (truths > lower_ci_riesz) * (truths < upper_ci_riesz)
            n_run = np.max(np.where(var_riesz > 0))
        else:
            covered_riesz = np.zeros(m)
            upper_ci_riesz = np.zeros(m)
            lower_ci_riesz = np.zeros(m)
            est_plugin_riesz = np.zeros(m)
            est_riesz = np.zeros(m)
            var_riesz = np.zeros(m)
            upper_ci_riesz = np.zeros(m)
            lower_ci_riesz = np.zeros(m)
            truths = np.zeros(m)
            n_run = 0

        for i in range(n_run, m):
            np.random.seed(i)
            torch.manual_seed(i)
            data = CData.load_chernozhukov_replication(i+1)
            truths[i] = data.get_average_treatment_effect()
            n = data.treatments.shape[0]
            n_features = data.covariates.shape[1]

            data = Dataset(np.concatenate([data.outcomes,data.treatments, data.covariates], axis = 1), outcome_column=0, treatment_column=1)

            network = ATERieszNetwork(
                ate_functional,
                features_in=n_features+1,
                hidden_shared=100,
                n_shared_layers=3,
                n_regression_weights=100,
                n_riesz_weights=100,
                n_regression_layers=2,
                n_riesz_layers = n_riesz
            )
            optim = Optimizer(network)
            loss = RieszNetLoss(rr_weight=rr_weight, tmle_weight=tmle_weight)
            riesz_net = RieszNetModule(network=network, loss=loss, optimizer=optim)
            riesz_net.fit(data)
            functional_riesz = riesz_net.get_functional(data).flatten()
            correction_riesz = riesz_net.get_correction(data).flatten()


            est_plugin_riesz[i] = np.mean(functional_riesz)

            est_riesz[i] = np.mean(est_plugin_riesz[i] + correction_riesz)
            var_riesz[i] = np.mean((functional_riesz - est_riesz[i] + correction_riesz) ** 2)
            lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
            upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
            covered_riesz[i] = (lower_ci_riesz[i] < truths[i]) * (truths[i] < upper_ci_riesz[i])

            print(f"RR weight {rr_weight} with TMLE, n_shared = {n}")
            print(f"RMSE : {np.sqrt(np.mean((est_riesz[:i+1]-truths[:i+1])**2))}, coverage = {np.mean(covered_riesz[:i+1])}, bias : {(np.mean((est_riesz[:i+1]-truths[:i+1])))}, MAE = {(np.mean(np.abs((est_riesz[:i+1]-truths[:i+1]))))}")
            print(i)

            headers = [
                "truth",
                "plugin_estimate_riesz",
                "riesz_estimate",
                "riesz_variance",
                "riesz_lower",
                "riesz_upper",
                "rr_weight"
            ]

            results = np.array(
                [
                    truths,
                    est_plugin_riesz,
                    est_riesz,
                    var_riesz,
                    lower_ci_riesz,
                    upper_ci_riesz,
                    [rr_weight for _ in range(m)]
                ]
            ).T

            np.savetxt(
                f"ate_experiment/neural_net_experiment/results/wTMLE/rr_w_{rr_weight}_TMLE_0.1.csv",
                results,
                delimiter=",",
                header=",".join(headers),
                comments="",
            )

if __name__ == "__main__":
    run(n_riesz=2)