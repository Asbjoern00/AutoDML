import numpy as np
from ate_experiment.dataset import Dataset
from RieszNet.DOPERieszNetModule import DOPERieszNetModule
from RieszNet.Optimizer import OptimizerParams
from RieszNet.DOPERieszNetATE import DOPEATERieszNetworkSimple
from average_treatment_effect.Functional.ATEFunctional import ate_functional
import torch

def run(n_shared):
    truth = 2.121539888279284
    n = 1000
    m = 1000
    n_folds = 5
    number_of_covariates = 10
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

        truths[i] = truth
        data = Dataset.simulate_dataset(n, number_of_covariates)
        folds = data.split_into_folds(n_folds)

        correction_riesz = np.zeros(data.treatments.shape[0])
        functional_riesz = np.zeros(data.treatments.shape[0])
        n_evaluated = 0
        for j in range(n_folds):
            eval_data, train_data = data.get_fit_and_train_folds(folds, j)
            n_eval_data = eval_data.treatments.shape[0]

            network = DOPEATERieszNetworkSimple(
                ate_functional,
                features_in=data.covariates.shape[1]+1,
                n_shared_layers=n_shared,
                n_regression_layers=2-n_shared,
                n_riesz_layers=2-n_shared,
                n_regression_weights=64,
                n_riesz_weights=64,
                hidden_shared=64,
                final_hidden_shared=64,
            )

            optim_regression = OptimizerParams(
                [network.shared, network.regression_treated, network.regression_untreated]
            )
            optim_rr = OptimizerParams([network.rr_head])

            riesz_net = DOPERieszNetModule(network=network, regression_optimizer=optim_regression, rr_optimizer=optim_rr)
            riesz_net.fit(data, informed="regression")

            functional_riesz[n_evaluated : n_evaluated + n_eval_data] = riesz_net.get_functional(eval_data).flatten()
            correction_riesz[n_evaluated : n_evaluated + n_eval_data] = riesz_net.get_correction(eval_data).flatten()

            n_evaluated = n_evaluated + n_eval_data

        est_plugin_riesz[i] = np.mean(functional_riesz)

        est_riesz[i] = np.mean(est_plugin_riesz[i] + correction_riesz)
        var_riesz[i] = np.mean((functional_riesz - est_riesz[i] + correction_riesz) ** 2)
        lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n_evaluated)
        upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n_evaluated)
        covered_riesz[i] = (lower_ci_riesz[i] < truths[i]) * (truths[i] < upper_ci_riesz[i])

        print(f"Outcome informed, n_shared = {n_shared}")
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
            [truth for _ in range(m)],
            est_plugin_riesz,
            est_riesz,
            var_riesz,
            lower_ci_riesz,
            upper_ci_riesz
        ]
     ).T

    np.savetxt(
        f"ate_experiment/neural_net_experiment/results/Dope/outcome_informed_n_shared_{n_shared}.csv",
       results,
      delimiter=",",
      header=",".join(headers),
       comments="",
    )
