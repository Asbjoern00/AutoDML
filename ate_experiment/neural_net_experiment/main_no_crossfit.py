import numpy as np
from ate_experiment.dataset import Dataset

from RieszNet.RieszNetModule import RieszNetModule,DragonNetModule
from RieszNet.Optimizer import Optimizer
from RieszNet.RieszNetATE import ATERieszNetwork
from RieszNet.Loss import RieszNetLoss,DragonNetLoss
from RieszNet.DragonNet import DragonNet
from average_treatment_effect.Functional.ATEFunctional import ate_functional

np.random.seed(1)

truth = 2.121539888279284
n = 1000
m = 1000
number_of_covariates = 10
rr_weight = 1.0
tmle_weight = 1.0
outcome_mse_weight = 1.0


est_plugin = np.zeros(m)

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
    network = ATERieszNetwork(ate_functional, features_in=number_of_covariates + 1)
    optim = Optimizer(network)
    loss = RieszNetLoss(rr_weight=rr_weight, tmle_weight=tmle_weight, outcome_mse_weight=outcome_mse_weight)
    riesz_net = RieszNetModule(network=network, loss=loss, optimizer=optim)
    riesz_net.fit(data)


    network = DragonNet(ate_functional, features_in=number_of_covariates)
    optim = Optimizer(network)
    loss = DragonNetLoss(rr_weight=rr_weight, tmle_weight=tmle_weight, outcome_mse_weight=outcome_mse_weight)
    propensity_net = DragonNetModule(network=network, loss=loss, optimizer=optim) #dragonnet
    propensity_net.fit(data)

    est_riesz[i] = riesz_net.get_double_robust(data)
    var_riesz[i] = np.mean((riesz_net.get_functional(data) - est_riesz[i] + riesz_net.get_correction(data)) ** 2)
    lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
    upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
    covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])
    print(f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}")

    est_propensity[i] = propensity_net.get_double_robust(data)
    var_propensity[i] = np.mean((propensity_net.get_functional(data) - est_propensity[i] + propensity_net.get_correction(data)) ** 2)
    lower_ci_propensity[i] = est_propensity[i] - 1.96 * np.sqrt(var_propensity[i] / n)
    upper_ci_propensity[i] = est_propensity[i] + 1.96 * np.sqrt(var_propensity[i] / n)
    covered_propensity[i] = (lower_ci_propensity[i] < truth) * (truth < upper_ci_propensity[i])
    print(f"Dragonnet MSE : {np.mean((est_propensity[:i+1]-truth)**2)}, coverage = {np.mean(covered_propensity[:i+1])}")
    print(i)


headers = [
    "truth",
    "plugin_estimate",
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
        est_plugin,
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
    "ate_experiment/neural_net_experiment/results/no_cross_fit_results.csv",
    results,
    delimiter=",",
    header=",".join(headers),
    comments="",
)