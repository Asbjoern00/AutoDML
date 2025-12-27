import numpy as np
from ase_experiment.dataset_highdim import DatasetHighDim
from LASSO.LassoClass import Lasso
from LASSO.RieszLasso import RieszLasso
from LASSO.OutcomeLASSO import OutcomeLASSO
from sklearn.linear_model import LassoCV
from ase_experiment.Functional.ASEFunctional import ase_functional
from ase_experiment.lasso_experiment.KDEestimator import ResidualKDE
import os.path


np.random.seed(1)

truth = 1.0
n = 1000
m = 1000
n_folds = 5


est_plugin = np.zeros(m)

est_riesz = np.zeros(m)
est_indirect = np.zeros(m)

var_riesz = np.zeros(m)
var_indirect = np.zeros(m)

covered_riesz = np.zeros(m)
covered_indirect = np.zeros(m)

upper_ci_riesz = np.zeros(m)
upper_ci_indirect = np.zeros(m)

lower_ci_riesz = np.zeros(m)
lower_ci_indirect = np.zeros(m)

if os.path.isfile("ase_experiment/lasso_experiment/Results/cross_fit_results.csv"):
    already_run = np.loadtxt(
    "ase_experiment/lasso_experiment/Results/cross_fit_results.csv",
    delimiter=",",
    skiprows=1
)
    n_already_run = already_run.shape[0]

    est_plugin[:n_already_run] = already_run[:, 1]

    est_indirect[:n_already_run] = already_run[:, 2]
    var_indirect[:n_already_run] = already_run[:, 3]
    lower_ci_indirect[:n_already_run] = already_run[:, 4]
    upper_ci_indirect[:n_already_run] = already_run[:, 5]

    est_riesz[:n_already_run] = already_run[:, 6]
    var_riesz[:n_already_run] = already_run[:, 7]
    lower_ci_riesz[:n_already_run] = already_run[:, 8]
    upper_ci_riesz[:n_already_run] = already_run[:, 9]
else:
    n_already_run = 0

for i in range(m):
    data = DatasetHighDim.simulate_dataset(n)
    folds = data.split_into_folds(n_folds)

    if i >= n_already_run or i == 8:
        correction_riesz = np.zeros(data.treatments.shape[0])
        correction_indirect = np.zeros(data.treatments.shape[0])
        functional = np.zeros(data.treatments.shape[0])
        n_evaluated = 0

        riesz_lasso = RieszLasso(ase_functional)
        indirect_riesz = ResidualKDE(model=LassoCV(fit_intercept=True, n_jobs=5, alphas = 10))
        outcome_lasso = OutcomeLASSO(ase_functional)
        lassoR = Lasso(riesz_lasso, outcome_lasso)
        lassoP = Lasso(indirect_riesz,outcome_lasso)

        for j in range(n_folds):
            eval_data, train_data = data.get_fit_and_train_folds(folds, j)
            n_eval_data = eval_data.treatments.shape[0]
            lassoR.fit(train_data)
            lassoP.fit(train_data)
            functional[n_evaluated : n_evaluated + n_eval_data] = lassoR.get_functional(eval_data)
            correction_riesz[n_evaluated : n_evaluated + n_eval_data] = lassoR.get_correction(eval_data)
            correction_indirect[n_evaluated : n_evaluated + n_eval_data] = lassoP.get_correction(eval_data)
            n_evaluated = n_evaluated + n_eval_data

        est_plugin[i] = np.mean(functional)



        est_riesz[i] = np.mean(est_plugin[i] + correction_riesz)
        var_riesz[i] = np.mean((functional - est_riesz[i] + correction_riesz) ** 2)
        lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
        upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)

        est_indirect[i] = np.mean(est_plugin[i] + correction_indirect)
        var_indirect[i] = np.mean((functional - est_indirect[i] + correction_indirect) ** 2)
        lower_ci_indirect[i] = est_indirect[i] - 1.96 * np.sqrt(var_indirect[i] / n)
        upper_ci_indirect[i] = est_indirect[i] + 1.96 * np.sqrt(var_indirect[i] / n)

    covered_indirect[i] = (lower_ci_indirect[i] < truth) * (truth < upper_ci_indirect[i])
    covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])

    print(f"Plugin MSE : {np.mean((est_plugin[:i+1]-truth)**2)}")

    print(f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}")

    print(
        f"indirect MSE : {np.mean((est_indirect[:i+1]-truth)**2)}, coverage = {np.mean(covered_indirect[:i+1])}"
    )
    print(i)


headers = [
    "truth",
    "plugin_estimate",
    "indirect_estimate",
    "indirect_variance",
    "indirect_lower",
    "indirect_upper",
    "riesz_estimate",
    "riesz_variance",
    "riesz_lower",
    "riesz_upper",
]

results = np.array(
    [
        [truth for _ in range(m)],
        est_plugin,
        est_indirect,
        var_indirect,
        lower_ci_indirect,
        upper_ci_indirect,
        est_riesz,
        var_riesz,
        lower_ci_riesz,
        upper_ci_riesz,
    ]
).T

np.savetxt(
    "ase_experiment/lasso_experiment/Results/cross_fit_results.csv",
    results,
    delimiter=",",
    header=",".join(headers),
    comments="",
)
