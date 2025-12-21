import numpy as np
from ate_experiment.dataset_highdim import DatasetHighDim
from average_treatment_effect.lasso.LassoClass import LassoATE
from average_treatment_effect.lasso.RieszLasso import RieszLasso, PropensityLasso
from average_treatment_effect.Functional.ATEFunctional import ate_functional

np.random.seed(1)

truth = 1.0
n = 1000
m = 1000
n_folds = 5


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
    data = DatasetHighDim.simulate_dataset(n)
    folds = data.split_into_folds(n_folds)

    correction_riesz = np.zeros_like(data.treatments.shape[0])
    correction_propensity = np.zeros_like(data.treatments.shape[0])
    functional = np.zeros_like(data.treatments.shape[0])
    n_evaluated = 0
    lassoR = LassoATE(RieszLasso)
    lassoP = LassoATE(PropensityLasso)

    for j in range(n_folds):
        eval_data, train_data = data.get_fit_and_train_folds(folds, j)
        n_eval_data = eval_data.treatments.shape[0]
        lassoR.fit(train_data)
        lassoP.fit(train_data)
        functional[n_evaluated : n_evaluated + n_eval_data] = lassoR.get_functional(eval_data)

        n_evaluated = n_evaluated + n_eval_data

    print(f"Plugin MSE : {np.mean((est_plugin[:i+1]-truth)**2)}")

    est_plugin[i] = lassoR.get_plugin(data)
    est_riesz[i] = lassoR.get_double_robust(data)
    var_riesz[i] = np.mean((lassoR.get_functional(data) - est_riesz[i] + lassoR.get_correction(data)) ** 2)
    lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
    upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
    covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])
    print(f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}")

    lassoP.fit(data)
    est_propensity[i] = lassoP.get_double_robust(data)
    var_propensity[i] = np.mean((lassoP.get_functional(data) - est_propensity[i] + lassoP.get_correction(data)) ** 2)
    lower_ci_propensity[i] = est_propensity[i] - 1.96 * np.sqrt(var_propensity[i] / n)
    upper_ci_propensity[i] = est_propensity[i] + 1.96 * np.sqrt(var_propensity[i] / n)
    covered_propensity[i] = (lower_ci_propensity[i] < truth) * (truth < upper_ci_propensity[i])
    print(
        f"Propensity MSE : {np.mean((est_propensity[:i+1]-truth)**2)}, coverage = {np.mean(covered_propensity[:i+1])}"
    )
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
    "ate_experiment/LASSO_experiment/Results/no_cross_fit_results.csv",
    results,
    delimiter=",",
    header=",".join(headers),
    comments="",
)
