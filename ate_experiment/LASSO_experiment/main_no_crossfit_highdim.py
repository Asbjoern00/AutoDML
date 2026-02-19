import numpy as np
from ate_experiment.dataset_highdim import DatasetHighDim
from LASSO.LassoClass import Lasso
from LASSO.OutcomeLASSO import OutcomeLASSO
from LASSO.RieszLasso import RieszLasso, PropensityLasso
from average_treatment_effect.Functional.ATEFunctional import ate_functional
import os
import pandas as pd


def run():
    ns = [125, 500, 2000]
    for n in ns:

        path = f"ate_experiment/LASSO_experiment/Results/no_cross_fit_results_{n}.csv"
        truth = 1.0
        m = 1000

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

        if os.path.exists(path):
            already_run = pd.read_csv(path)
            n_already_run = np.argmax(already_run["riesz_variance"] == 0)

            est_plugin[:n_already_run] = already_run["plugin_estimate"][:n_already_run]
            est_propensity[:n_already_run] = already_run["propensity_estimate"][:n_already_run]
            var_propensity[:n_already_run] = already_run["propensity_variance"][:n_already_run]
            lower_ci_propensity[:n_already_run] = already_run["propensity_lower"][:n_already_run]
            upper_ci_propensity[:n_already_run] = already_run["propensity_upper"][:n_already_run]
            est_riesz[:n_already_run] = already_run["riesz_estimate"][:n_already_run]
            var_riesz[:n_already_run] = already_run["riesz_variance"][:n_already_run]
            lower_ci_riesz[:n_already_run] = already_run["riesz_lower"][:n_already_run]
            upper_ci_riesz[:n_already_run] = already_run["riesz_upper"][:n_already_run]
            covered_propensity[:n_already_run] = (lower_ci_propensity[:n_already_run] < truth) * (
                truth < upper_ci_propensity[:n_already_run]
            )
            covered_riesz[:n_already_run] = (lower_ci_riesz[:n_already_run] < truth) * (
                truth < upper_ci_riesz[:n_already_run]
            )

        else:
            n_already_run = 0

        for i in range(n_already_run, m):
            np.random.seed(i)
            data = DatasetHighDim.simulate_dataset(n)

            outcome_lasso = OutcomeLASSO(ate_functional)
            riesz_lasso = RieszLasso(ate_functional, expand_treatment=True)
            propensity_lasso = PropensityLasso()

            lassoR = Lasso(riesz_lasso, outcome_lasso)
            lassoR.fit(data, cv_riesz_c1s=np.array([5 / 4, 3 / 4, 2 / 3, 1 / 2]))

            est_plugin[i] = lassoR.get_plugin(data)
            print(
                f"Plugin MSE : {np.mean((est_plugin[:i+1]-truth)**2)}, scaled bias :{np.sqrt(n)*np.mean((est_plugin[:i+1]-truth))}"
            )

            est_riesz[i] = lassoR.get_double_robust(data)
            var_riesz[i] = np.mean((lassoR.get_functional(data) - est_riesz[i] + lassoR.get_correction(data)) ** 2)
            lower_ci_riesz[i] = est_riesz[i] - 1.96 * np.sqrt(var_riesz[i] / n)
            upper_ci_riesz[i] = est_riesz[i] + 1.96 * np.sqrt(var_riesz[i] / n)
            covered_riesz[i] = (lower_ci_riesz[i] < truth) * (truth < upper_ci_riesz[i])
            print(
                f"Riesz MSE : {np.mean((est_riesz[:i+1]-truth)**2)}, coverage = {np.mean(covered_riesz[:i+1])}, scaled bias :{np.sqrt(n)*np.mean((est_riesz[:i+1]-truth))}"
            )

            lassoP = Lasso(propensity_lasso, outcome_lasso)
            lassoP.fit(data, fit_outcome_model=False)
            est_propensity[i] = lassoP.get_double_robust(data)
            var_propensity[i] = np.mean(
                (lassoP.get_functional(data) - est_propensity[i] + lassoP.get_correction(data)) ** 2
            )
            lower_ci_propensity[i] = est_propensity[i] - 1.96 * np.sqrt(var_propensity[i] / n)
            upper_ci_propensity[i] = est_propensity[i] + 1.96 * np.sqrt(var_propensity[i] / n)
            covered_propensity[i] = (lower_ci_propensity[i] < truth) * (truth < upper_ci_propensity[i])
            print(
                f"Propensity MSE : {np.mean((est_propensity[:i+1]-truth)**2)}, coverage = {np.mean(covered_propensity[:i+1])}, scaled bias :{np.sqrt(n)*np.mean((est_propensity[:i+1]-truth))}"
            )

            print(i, n)

            headers = [
                "truth",
                "n",
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
                    [n for _ in range(m)],
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
                path,
                results,
                delimiter=",",
                header=",".join(headers),
                comments="",
            )


if __name__ == "__main__":
    run()
