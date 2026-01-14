import numpy as np
import torch
from dope_neural_nets.model import ModelWrapper
from dope_neural_nets.dataset import Dataset
from RieszNet.RieszNetModule import RieszNetModule
from RieszNet.Optimizer import Optimizer
from RieszNet.RieszNetATE import ATERieszNetworkSimple
from RieszNet.Loss import RieszNetLoss
from average_treatment_effect.Functional.ATEFunctional import ate_functional


truth = 2.121539888279284
n = 1000
m = 1000
n_folds = 5
number_of_covariates = 10
rr_weights = 2.0**np.arange(start = -5, stop = 5, step = 1)
#rr_weights = [10.0]
tmle_weight = 0.0


np.random.seed(42)
torch.manual_seed(42)

for rr_weight in rr_weights:

    est_plugin_riesz = np.zeros(m)

    est_riesz = np.zeros(m)

    var_riesz = np.zeros(m)

    covered_riesz = np.zeros(m)

    upper_ci_riesz = np.zeros(m)

    lower_ci_riesz = np.zeros(m)

    for i in range(m):
        data = Dataset.simulate_dataset(n, number_of_covariates)
        folds = data.split_into_folds(n_folds)

        correction_riesz = np.zeros(data.treatments.shape[0])
        correction_propensity = np.zeros(data.treatments.shape[0])
        functional_riesz = np.zeros(data.treatments.shape[0])
        functional_propensity = np.zeros(data.treatments.shape[0])
        n_evaluated = 0
        estimate_components = []
        for j in range(n_folds):
            eval_data, train_data = data.get_fit_and_train_folds(folds, j)
            n_eval_data = eval_data.treatments.shape[0]

            network = ATERieszNetworkSimple(ate_functional, features_in=number_of_covariates + 1)
            optim = Optimizer(network)
            loss = RieszNetLoss(rr_weight=rr_weight, tmle_weight=tmle_weight)
            riesz_net = RieszNetModule(network=network, loss=loss, optimizer=optim)

            riesz_net.fit(data)

            functional_riesz[n_evaluated : n_evaluated + n_eval_data] = riesz_net.get_functional(eval_data).flatten()
            correction_riesz[n_evaluated : n_evaluated + n_eval_data] = riesz_net.get_correction(eval_data).flatten()

            n_evaluated = n_evaluated + n_eval_data
#            model_wrapper = ModelWrapper()
#            model_wrapper.train_as_riesz_net(train_data, rr_w=rr_weight)
#            estimate_components.append(model_wrapper.get_estimate_components(eval_data))

#        estimate_components = torch.concat(estimate_components, dim=0)
#        estimate = torch.mean(estimate_components).item()
#        variance = torch.var(estimate_components).item()

        est_plugin_riesz[i] = np.mean(functional_riesz)

        est_riesz[i] = np.mean(est_plugin_riesz[i] + correction_riesz)
        var_riesz[i] = np.mean((functional_riesz - est_riesz[i] + correction_riesz) ** 2)
        lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
        upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
        covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])

        print(
            f"Riesz weight = {rr_weight} RMSE : {np.sqrt(np.mean((est_riesz[:i+1]-truth)**2))}, coverage = {np.mean(covered_riesz[:i+1])}, bias = {np.mean((est_riesz[:i+1]-truth))},var = {np.var(est_riesz[:i+1])}"
        )
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
            [truth for _ in range(m)],
            est_plugin_riesz,
            est_riesz,
            var_riesz,
            lower_ci_riesz,
            upper_ci_riesz,
            [rr_weight for _ in range(m)]
        ]
    ).T

    np.savetxt(
        f"ate_experiment/neural_net_experiment/results/woTMLE/rr_w_{rr_weight}.csv",
        results,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
