import numpy as np
from ate_experiment.dataset_highdim import DatasetHighDim
from LASSO.LassoClass import Lasso, OutcomeAdaptedLasso
from LASSO.OutcomeLASSO import OutcomeLASSO
from LASSO.RieszLasso import RieszLasso
from average_treatment_effect.Functional.ATEFunctional import ate_functional


def run():
    path = "ate_experiment/LASSO_experiment/DOPE/InformedLasso.csv"
    propensity_coef_file = "ate_experiment/LASSO_experiment/propensity_coefficients.npy"
    regression_coef_file = "ate_experiment/LASSO_experiment/regression_coefficients.npy"
    propensity_beta = np.load(propensity_coef_file)
    outcome_beta = np.load(regression_coef_file)

    active_propensity = np.nonzero(propensity_beta)[0]

    active_regression = np.nonzero(outcome_beta)[0] - 1
    active_regression = active_regression[active_regression > 0]
    confounders = np.intersect1d(active_regression, active_propensity)

    truth = 1.0
    n = 500
    m = 1000

    est_riesz = np.zeros(m)
    est_riesz_oracle_propensity = np.zeros(m)
    est_riesz_outcome_adapted = np.zeros(m)
    est_riesz_oracle_outcome = np.zeros(m)
    est_riesz_informed = np.zeros(m)
    covered_outcome_adapted = np.zeros(m)
    covered_riesz_adapted = np.zeros(m)

    for i in range(m):
        np.random.seed(i)
        data = DatasetHighDim.simulate_dataset(n)

        outcome_lasso = OutcomeLASSO(ate_functional)
        outcome_lasso_riesz_informed = OutcomeLASSO(ate_functional)
        outcome_lasso_riesz_oracle = OutcomeLASSO(ate_functional)

        riesz_lasso_outcome_adapted = RieszLasso(ate_functional, expand_treatment=True)
        riesz_lasso_oracle_propensity = RieszLasso(ate_functional, expand_treatment=True)
        riesz_lasso_oracle_outcome = RieszLasso(ate_functional, expand_treatment=True)
        riesz_lasso = RieszLasso(ate_functional, expand_treatment=True)

        riesz_lasso_oracle_outcome.set_covariate_indices(active_regression)
        outcome_lasso_riesz_oracle.set_covariate_indices(active_propensity)

        lassoOA = OutcomeAdaptedLasso(riesz_lasso_outcome_adapted, outcome_lasso)
        lassoOutcome = Lasso(riesz_lasso_oracle_outcome, outcome_lasso)
        lassoPropensity = Lasso(riesz_lasso_oracle_propensity, outcome_lasso_riesz_oracle)
        lasso = Lasso(riesz_lasso, outcome_lasso)

        cv_riesz_c1s = np.array([8, 4, 2, 5 / 4, 3 / 4, 2 / 3, 1 / 2])

        lassoOA.fit(data, cv_riesz_c1s=cv_riesz_c1s)
        lassoOutcome.fit(data, cv_riesz_c1s=cv_riesz_c1s, fit_outcome_model=False)
        lassoPropensity.fit(data, cv_riesz_c1s=cv_riesz_c1s, fit_outcome_model=True)
        lasso.fit(data, cv_riesz_c1s=cv_riesz_c1s, fit_outcome_model=False)

        active_riesz = np.where(riesz_lasso.rho > 0)[0] - 2
        active_riesz = active_riesz[active_riesz > 0]
        outcome_lasso_riesz_informed.set_covariate_indices(active_riesz)
        outcome_lasso_riesz_informed.fit(data)

        covered_outcome_adapted[i] = (
            np.intersect1d(confounders, outcome_lasso.get_active_covariate_indices()).shape[0]
        ) / (confounders.shape[0])
        covered_riesz_adapted[i] = (
            np.intersect1d(confounders, active_riesz).shape[0]
        ) / (confounders.shape[0])

        lassoRA = Lasso(riesz_lasso, outcome_lasso_riesz_informed)

        est_riesz_outcome_adapted[i] = lassoOA.get_double_robust(data)
        est_riesz_oracle_propensity[i] = lassoPropensity.get_double_robust(data)
        est_riesz_oracle_outcome[i] = lassoOutcome.get_double_robust(data)
        est_riesz[i] = lasso.get_double_robust(data)
        est_riesz_informed[i] = lassoRA.get_double_robust(data)

        print(
            f"Outcome RMSE : {np.sqrt(np.mean((est_riesz_oracle_outcome[:i+1]-truth)**2))}, Scaled bias = {(np.mean((est_riesz_oracle_outcome[:i+1]-truth)))}"
        )
        print(
            f"Outcome Adapted RMSE : {np.sqrt(np.mean((est_riesz_outcome_adapted[:i+1]-truth)**2))}, Scaled bias = {(np.mean((est_riesz_outcome_adapted[:i+1]-truth)))}, covered = {np.mean(covered_outcome_adapted[:i+1])}"
        )

        print(
            f"Propensity RMSE : {np.sqrt(np.mean((est_riesz_oracle_propensity[:i+1]-truth)**2))}, Scaled bias = {(np.mean((est_riesz_oracle_propensity[:i+1]-truth)))}"
        )
        print(
            f"Riesz RMSE : {np.sqrt(np.mean((est_riesz[:i+1]-truth)**2))}, Scaled bias = {(np.mean((est_riesz[:i+1]-truth)))}"
        )
        print(
            f"Riesz Informed RMSE : {np.sqrt(np.mean((est_riesz_informed[:i+1]-truth)**2))}, Scaled bias = {(np.mean((est_riesz_informed[:i+1]-truth)))}, covered = {np.mean(covered_riesz_adapted[:i+1])}"
        )
        print(i)

        headers = ["truth", "Riesz_Informed", "Riesz_Oracle", "Riesz", "Outcome_Informed", "Outcome_Oracle", "covered_outcome_adapted", "covered_riesz_adapted"]

        results = np.array(
            [
                [truth for _ in range(m)],
                est_riesz_informed,
                est_riesz_oracle_propensity,
                est_riesz,
                est_riesz_outcome_adapted,
                est_riesz_oracle_outcome,
                covered_outcome_adapted,
                covered_riesz_adapted,
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
