import numpy as np
from ate_experiment.dataset import Dataset

from RieszNet.RieszNetModule import RieszNetModule, DragonNetModule
from RieszNet.Optimizer import Optimizer
from RieszNet.RieszNetATE import ATERieszNetwork
from RieszNet.DragonNet import DragonNet
from RieszNet.Loss import RieszNetLoss, DragonNetLoss
from average_treatment_effect.Functional.ATEFunctional import ate_functional

np.random.seed(1)

truth = 2.121539888279284
n = 1000
m = 1000
n_folds = 5
number_of_covariates = 10
rr_weight = 1.0
tmle_weight = 1.0
outcome_mse_weight = 1.0


est_plugin_riesz = np.zeros(m)
est_plugin_propensity = np.zeros(m)

est_riesz = np.zeros(m)
est_propensity = np.zeros(m)

var_riesz = np.zeros(m)
var_propensity = np.zeros(m)

covered_riesz = np.zeros(m)
covered_propensity = np.zeros(m)

upper_ci_riesz = np.zeros(m)
upper_ci_propensity = np.zeros(m)

lower_ci_riesz = np.zeros(m)
lower_ci_propensity = np.zeros(m)

for i in range(m):
    data = Dataset.simulate_dataset(n, number_of_covariates)
    folds = data.split_into_folds(n_folds)

    correction_riesz = np.zeros(data.treatments.shape[0])
    correction_propensity = np.zeros(data.treatments.shape[0])Neural
    functional_riesz = np.zeros(data.treatments.shape[0])
    functional_propensity = np.zeros(data.treatments.shape[0])

    n_evaluated = 0

    for j in range(n_folds):
        eval_data, train_data = data.get_fit_and_train_folds(folds, j)
        n_eval_data = eval_data.treatments.shape[0]

        network = ATERieszNetwork(ate_functional, features_in=number_of_covariates + 1)
        optim = Optimizer(network)
        loss = RieszNetLoss(rr_weight=rr_weight, tmle_weight=tmle_weight, outcome_mse_weight=outcome_mse_weight)
        riesz_net = RieszNetModule(network=network, loss=loss, optimizer=optim)

        network = DragonNet(ate_functional, features_in=number_of_covariates)
        optim = Optimizer(network)
        loss = DragonNetLoss(rr_weight=rr_weight, tmle_weight=tmle_weight, outcome_mse_weight=outcome_mse_weight)
        propensity_net = DragonNetModule(network=network, loss=loss, optimizer=optim)  # dragonnet

        riesz_net.fit(data)
        propensity_net.fit(data)

        functional_riesz[n_evaluated : n_evaluated + n_eval_data] = riesz_net.get_functional(eval_data).flatten()
        correction_riesz[n_evaluated : n_evaluated + n_eval_data] = riesz_net.get_correction(eval_data).flatten()

        functional_propensity[n_evaluated : n_evaluated + n_eval_data] = propensity_net.get_functional(eval_data).flatten()
        correction_propensity[n_evaluated : n_evaluated + n_eval_data] = propensity_net.get_correction(eval_data).flatten()
        n_evaluated = n_evaluated + n_eval_data

    est_plugin_riesz[i] = np.mean(functional_riesz)
    est_plugin_propensity[i] = np.mean(functional_propensity)

    est_riesz[i] = np.mean(est_plugin_riesz[i] + correction_riesz)
    var_riesz[i] = np.mean((functional_riesz - est_riesz[i] + correction_riesz) ** 2)
    lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
    upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)

    est_propensity[i] = np.mean(est_plugin_propensity[i] + correction_propensity)
    var_propensity[i] = np.mean((functional_propensity - est_propensity[i] + correction_propensity) ** 2)
    lower_ci_propensity[i] = est_propensity[i] - 1.96 * np.sqrt(var_propensity[i] / n)
    upper_ci_propensity[i] = est_propensity[i] + 1.96 * np.sqrt(var_propensity[i] / n)

    covered_propensity[i] = (lower_ci_propensity[i] < truth) * (truth < upper_ci_propensity[i])
    covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])

    print(f"Plugin Riesz MSE : {np.mean((est_plugin_riesz[:i+1]-truth)**2)}")
    print(f"Plugin Propensity MSE : {np.mean((est_plugin_riesz[:i+1]-truth)**2)}")

    print(f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}")

    print(
        f"Propensity MSE : {np.mean((est_propensity[:i+1]-truth)**2)}, coverage = {np.mean(covered_propensity[:i+1])}"
    )
    print(i)


headers = [
    "truth",
    "plugin_estimate_riesz",
    "plugin_estimate_propensity",
    "propensity_estimate",
    "propensity_variance",
    "propensity_lower",
    "propensity_upper",
    "riesz_estimate",
    "riesz_variance",
    "riesz_lower",
    "riesz_upper",
]

results = np.array(
    [
        [truth for _ in range(m)],
        est_plugin_riesz,
        est_plugin_propensity,
        est_propensity,
        var_propensity,
        lower_ci_propensity,
        upper_ci_propensity,
        est_riesz,
        var_riesz,
        lower_ci_riesz,
        upper_ci_riesz,
    ]
).T

np.savetxt(
    "ate_experiment/neural_net_experiment/results/cross_fit_results.csv",
    results,
    delimiter=",",
    header=",".join(headers),
    comments="",
)
