import numpy as np
from ate_experiment.dataset_highdim import DatasetHighDim
from LASSO.LassoClass import Lasso,OutcomeAdaptedLasso
from LASSO.OutcomeLASSO import OutcomeLASSO
from LASSO.RieszLasso import RieszLasso, PropensityLasso
import os.path
from average_treatment_effect.Functional.ATEFunctional import ate_functional


np.random.seed(1)

truth = 1.0
n = 1000
m = 1000
n_folds = 5


est_plugin = np.zeros(m)

est_riesz = np.zeros(m)
est_propensity = np.zeros(m)
est_riesz_oa = np.zeros(m)

var_riesz = np.zeros(m)
var_propensity = np.zeros(m)
var_riesz_oa = np.zeros(m)

covered_riesz = np.zeros(m)
covered_propensity = np.zeros(m)
covered_riesz_oa = np.zeros(m)

upper_ci_riesz = np.zeros(m)
upper_ci_propensity = np.zeros(m)
upper_ci_riesz_oa = np.zeros(m)

lower_ci_riesz = np.zeros(m)
lower_ci_propensity = np.zeros(m)
lower_ci_riesz_oa = np.zeros(m)


n_already_run = 0

for i in range(m):
    data = DatasetHighDim.simulate_dataset(n,"ate_experiment/LASSO_experiment/propensity_coefficients.npy",
                                           "ate_experiment/LASSO_experiment/regression_coefficients.npy")
    folds = data.split_into_folds(n_folds)

    if i >= n_already_run:
        correction_riesz = np.zeros(data.treatments.shape[0])
        correction_propensity = np.zeros(data.treatments.shape[0])
        correction_riesz_oa = np.zeros(data.treatments.shape[0])

        functional_riesz = np.zeros(data.treatments.shape[0])
        functional_propensity = np.zeros(data.treatments.shape[0])
        functional_riesz_oa = np.zeros(data.treatments.shape[0])

        n_evaluated = 0

        riesz_lasso = RieszLasso(ate_functional)
        riesz_lasso_oa = RieszLasso(ate_functional)
        propensity_lasso = PropensityLasso()
        outcome_lasso = OutcomeLASSO(ate_functional)
        lassoR = Lasso(riesz_lasso, outcome_lasso)
        lassoP = Lasso(propensity_lasso,outcome_lasso)
        lassoRoa = OutcomeAdaptedLasso(riesz_lasso_oa, outcome_lasso)

        for j in range(n_folds):
            eval_data, train_data = data.get_fit_and_train_folds(folds, j)
            n_eval_data = eval_data.treatments.shape[0]
            lassoR.fit(train_data)
            lassoP.fit(train_data,fit_outcome_model=False)
            lassoRoa.fit(train_data, fit_outcome_model=False)

            functional_riesz[n_evaluated : n_evaluated + n_eval_data] = lassoR.get_functional(eval_data)
            functional_propensity[n_evaluated : n_evaluated + n_eval_data] = lassoP.get_functional(eval_data)
            functional_riesz_oa[n_evaluated : n_evaluated + n_eval_data] = lassoRoa.get_functional(eval_data)

            correction_riesz[n_evaluated : n_evaluated + n_eval_data] = lassoR.get_correction(eval_data)
            correction_riesz_oa[n_evaluated : n_evaluated + n_eval_data] = lassoRoa.get_correction(eval_data)
            correction_propensity[n_evaluated : n_evaluated + n_eval_data] = lassoP.get_correction(eval_data)

            n_evaluated = n_evaluated + n_eval_data

        est_plugin[i] = np.mean(functional_riesz)

        est_riesz[i] = np.mean(est_plugin[i] + correction_riesz)
        var_riesz[i] = np.mean((functional_riesz - est_riesz[i] + correction_riesz) ** 2)
        lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
        upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)

        est_propensity[i] = np.mean(est_plugin[i] + correction_propensity)
        var_propensity[i] = np.mean((functional_propensity - est_propensity[i] + correction_propensity) ** 2)
        lower_ci_propensity[i] = est_propensity[i] - 1.96 * np.sqrt(var_propensity[i] / n)
        upper_ci_propensity[i] = est_propensity[i] + 1.96 * np.sqrt(var_propensity[i] / n)

        est_riesz_oa[i] = np.mean(est_plugin[i] + correction_riesz_oa)
        var_riesz_oa[i] = np.mean((functional_riesz_oa - est_riesz_oa[i] + correction_riesz_oa) ** 2)
        lower_ci_riesz_oa[i] = est_riesz_oa[i] - 1.96 * np.sqrt(var_riesz_oa[i] / n)
        upper_ci_riesz_oa[i] = est_riesz_oa[i] + 1.96 * np.sqrt(var_riesz_oa[i] / n)

    covered_propensity[i] = (lower_ci_propensity[i] < truth) * (truth < upper_ci_propensity[i])
    covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])
    covered_riesz_oa[i] = (lower_ci_riesz_oa[i] < truth) * (truth < upper_ci_riesz_oa[i])

    print(f"Plugin MSE : {np.mean((est_plugin[:i+1]-truth)**2)}")

    print(f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}")

    print(
        f"Propensity MSE : {np.mean((est_propensity[:i+1]-truth)**2)}, coverage = {np.mean(covered_propensity[:i+1])}"
    )
    print(
        f"Riesz OA MSE : {np.mean((est_riesz_oa[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz_oa[:i+1])}"
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
        "ate_experiment/LASSO_experiment/Results/cross_fit_results.csv",
        results,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
