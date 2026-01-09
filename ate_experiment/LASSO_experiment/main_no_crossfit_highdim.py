import numpy as np
from ate_experiment.dataset_highdim import DatasetHighDim
from LASSO.LassoClass import Lasso, OutcomeAdaptedLasso
from LASSO.OutcomeLASSO import OutcomeLASSO
from LASSO.RieszLasso import RieszLasso, PropensityLasso
from average_treatment_effect.Functional.ATEFunctional import ate_functional

np.random.seed(1)

truth = 1.0
n = 1000
m = 1000


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


for i in range(m):
    data = DatasetHighDim.simulate_dataset(n)

    outcome_lasso = OutcomeLASSO(ate_functional)
    riesz_lasso = RieszLasso(ate_functional)
    propensity_lasso = PropensityLasso()
    riesz_lasso_oa = RieszLasso(ate_functional)

    lassoR = Lasso(riesz_lasso, outcome_lasso)
    lassoR.fit(data, cv_riesz_c1s=np.array([1.25, 1, 0.75, 0.5, 0.25]))

    lassoP = Lasso(propensity_lasso, outcome_lasso)
    lassoP.fit(data, fit_outcome_model=False)

    lassoRoa = OutcomeAdaptedLasso(riesz_lasso_oa, outcome_lasso)
    lassoRoa.fit(data, cv_riesz_c1s=np.array([1.25, 1, 0.75, 0.5, 0.25]), fit_outcome_model=False)

    est_plugin[i] = lassoR.get_plugin(data)
    print(f"Plugin MSE : {np.mean((est_plugin[:i+1]-truth)**2)}, bias :{np.sqrt(n)*np.mean((est_plugin[:i+1]-truth))}")

    est_riesz[i] = lassoR.get_double_robust(data)
    var_riesz[i] = np.mean((lassoR.get_functional(data) - est_riesz[i] + lassoR.get_correction(data)) ** 2)
    lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
    upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
    covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])
    print(
        f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}, bias :{np.sqrt(n)*np.mean((est_riesz[:i+1]-truth))}"
    )

    lassoP = Lasso(propensity_lasso, outcome_lasso)
    lassoP.fit(data)
    est_propensity[i] = lassoP.get_double_robust(data)
    var_propensity[i] = np.mean((lassoP.get_functional(data) - est_propensity[i] + lassoP.get_correction(data)) ** 2)
    lower_ci_propensity[i] = est_propensity[i] - 1.96 * np.sqrt(var_propensity[i] / n)
    upper_ci_propensity[i] = est_propensity[i] + 1.96 * np.sqrt(var_propensity[i] / n)
    covered_propensity[i] = (lower_ci_propensity[i] < truth) * (truth < upper_ci_propensity[i])
    print(
        f"Propensity MSE : {np.mean((est_propensity[:i+1]-truth)**2)}, coverage = {np.mean(covered_propensity[:i+1])}, bias :{np.sqrt(n)*np.mean((est_propensity[:i+1]-truth))}"
    )

    est_riesz_oa[i] = lassoRoa.get_double_robust(data)
    var_riesz_oa[i] = np.mean((lassoRoa.get_functional(data) - est_riesz_oa[i] + lassoRoa.get_correction(data)) ** 2)
    lower_ci_riesz_oa[i] = est_riesz_oa[i] - 1.96 * np.sqrt(var_riesz_oa[i] / n)
    upper_ci_riesz_oa[i] = est_riesz_oa[i] + 1.96 * np.sqrt(var_riesz_oa[i] / n)
    covered_riesz_oa[i] = (lower_ci_riesz_oa[i] < truth) * (truth < upper_ci_riesz_oa[i])
    print(f"Riesz OA MSE : {np.mean((est_riesz_oa[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz_oa[:i+1])}, bias :{np.sqrt(n)*np.mean((est_riesz_oa[:i+1]-truth))}")

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
        "riesz_oa_estimate",
        "riesz_oa_variance",
        "riesz_oa_lower",
        "riesz_oa_upper",
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
            est_riesz_oa,
            var_riesz_oa,
            lower_ci_riesz_oa,
            upper_ci_riesz_oa,
        ]
    ).T

    np.savetxt(
        "ate_experiment/LASSO_experiment/Results/no_cross_fit_results.csv",
        results,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
