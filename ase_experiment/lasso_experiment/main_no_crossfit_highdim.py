import numpy as np
from ase_experiment.dataset_highdim import DatasetHighDim
from LASSO.LassoClass import Lasso
from LASSO.RieszLasso import RieszLasso,ASETreatmentLasso
from LASSO.OutcomeLASSO import OutcomeLASSO
from ase_experiment.Functional.ASEFunctional import ase_functional

np.random.seed(1)

truth = 1.0
n = 1000
m = 1000


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


for i in range(m):
    data = DatasetHighDim.simulate_dataset(n)

    riesz_lasso = RieszLasso(ase_functional)
    indirect_riesz = ASETreatmentLasso()
    outcome_lasso = OutcomeLASSO(ase_functional)

    lassoR = Lasso(riesz_lasso, outcome_lasso)
    lassoR.fit(data, cv_riesz_c1s=np.array([1.25, 1, 0.75, 0.5, 0.25]))

    est_plugin[i] = lassoR.get_plugin(data)
    print(f"Plugin MSE : {np.mean((est_plugin[:i+1]-truth)**2)}")

    est_riesz[i] = lassoR.get_double_robust(data)
    var_riesz[i] = np.mean((lassoR.get_functional(data) - est_riesz[i] + lassoR.get_correction(data)) ** 2)
    lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
    upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
    covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])
    print(f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}")

    lassoP = Lasso(indirect_riesz, outcome_lasso)
    lassoP.fit(data, fit_outcome_model=False)
    est_indirect[i] = lassoP.get_double_robust(data)
    var_indirect[i] = np.mean((lassoP.get_functional(data) - est_indirect[i] + lassoP.get_correction(data)) ** 2)
    lower_ci_indirect[i] = est_indirect[i] - 1.96 * np.sqrt(var_indirect[i] / n)
    upper_ci_indirect[i] = est_indirect[i] + 1.96 * np.sqrt(var_indirect[i] / n)
    covered_indirect[i] = (lower_ci_indirect[i] < truth) * (truth < upper_ci_indirect[i])
    print(f"indirect MSE : {np.mean((est_indirect[:i+1]-truth)**2)}, coverage = {np.mean(covered_indirect[:i+1])}")
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
        "ase_experiment/lasso_experiment/Results/no_cross_fit_results.csv",
        results,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
