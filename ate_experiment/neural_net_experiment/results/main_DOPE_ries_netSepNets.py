import numpy as np
from ihdp_average_treatment_effect.dataset import Dataset as CData
from ate_experiment.dataset import Dataset
from RieszNet.DOPERieszNetModule import DOPERieszNetModule
from RieszNet.Optimizer import OptimizerParams
from RieszNet.DOPERieszNetATE import DOPEATERieszNetworkNonShared
from average_treatment_effect.Functional.ATEFunctional import ate_functional
import torch

def run():
    m = 1000
    est_plugin_riesz = np.zeros(m)

    est_riesz = np.zeros(m)

    var_riesz = np.zeros(m)

    covered_riesz = np.zeros(m)

    upper_ci_riesz = np.zeros(m)

    lower_ci_riesz = np.zeros(m)

    truths = np.zeros(m)

    for i in range(m):
        np.random.seed(i)
        torch.manual_seed(i)
        data = CData.load_chernozhukov_replication(i+1)
        truths[i] = data.get_average_treatment_effect()
        n = data.treatments.shape[0]
        n_features = data.covariates.shape[1]

        data = Dataset(np.concatenate([data.outcomes,data.treatments, data.covariates], axis = 1), outcome_column=0, treatment_column=1)

        network = DOPEATERieszNetworkNonShared(
            ate_functional,
            features_in=n_features+1,
            n_regression_layers=2,
            n_riesz_layers=5,
            shared_regression=3,
            n_regression_weights=100,
            n_riesz_weights=100
        )

        optim_regression = OptimizerParams(
            [network.regression_treated,network.regression_untreated,network.shared_regression]
        )
        optim_rr = OptimizerParams([network.rr])

        riesz_net = DOPERieszNetModule(network=network, regression_optimizer=optim_regression, rr_optimizer=optim_rr)
        riesz_net.fit(data, informed="separate")

        functional_riesz = riesz_net.get_functional(data).flatten()
        correction_riesz = riesz_net.get_correction(data).flatten()

        est_plugin_riesz[i] = np.mean(functional_riesz)

        est_riesz[i] = np.mean(est_plugin_riesz[i] + correction_riesz)
        var_riesz[i] = np.mean((functional_riesz - est_riesz[i] + correction_riesz) ** 2)
        lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
        upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
        covered_riesz[i] = (lower_ci_riesz[i] < truths[i]) * (truths[i] < upper_ci_riesz[i])

        print(f"RMSE : {np.sqrt(np.mean((est_riesz[:i+1]-truths[:i+1])**2))}, coverage = {np.mean(covered_riesz[:i+1])}, bias : {(np.mean((est_riesz[:i+1]-truths[:i+1])))}")
        print(i)

        headers = [
            "truth",
            "plugin_estimate_riesz",
            "riesz_estimate",
            "riesz_variance",
            "riesz_lower",
            "riesz_upper"
        ]

        results = np.array(
            [
                truths,
                est_plugin_riesz,
                est_riesz,
                var_riesz,
                lower_ci_riesz,
                upper_ci_riesz
            ]
         ).T

        np.savetxt(
            f"ate_experiment/neural_net_experiment/results/Dope/sep_nets.csv",
           results,
          delimiter=",",
          header=",".join(headers),
           comments="",
        )
if __name__ == "__main__":
    run()
