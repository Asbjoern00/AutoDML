import xgboost as xgb
import numpy as np

from average_treatment_effect.dataset import Dataset

folds = 10
ests = []
truths = []

for i in range(1000):
    dr_est = 0
    data = Dataset.load_chernozhukov_replication(i + 1)
    truth = data.get_average_treatment_effect()
    truths.append(truth)
    treatments = data.treatments[:, 0]
    outcomes = data.outcomes[:, 0]
    covariates = data.covariates
    outcome_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "eta": 0.1, "max_depth": 3, "lambda": 3}
    outcome_model0 = xgb.train(
        outcome_params,
        xgb.DMatrix(covariates[treatments == 0, :], label=outcomes[treatments == 0]),
        num_boost_round=1000,
        evals=[(xgb.DMatrix(covariates[treatments == 0, :], label=outcomes[treatments == 0]), "train")],
        verbose_eval=0,
    )
    outcome_model1 = xgb.train(
        outcome_params,
        xgb.DMatrix(covariates[treatments == 1, :], label=outcomes[treatments == 1]),
        num_boost_round=1000,
        evals=[(xgb.DMatrix(covariates, label=outcomes), "train")],
        verbose_eval=0,
    )

    treatment_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 3,
        "lambda": 3,
    }
    treatment_model = xgb.train(
        treatment_params,
        xgb.DMatrix(covariates, label=treatments),
        num_boost_round=1000,
        evals=[((xgb.DMatrix(covariates, label=treatments)), "train")],
        verbose_eval=0,
    )

    q1 = outcome_model1.predict(xgb.DMatrix(covariates, label=outcomes))
    q0 = outcome_model0.predict(xgb.DMatrix(covariates, label=outcomes))
    plugin = np.mean(q1 - q0)
    residuals = outcomes - (treatments * q1 + (1 - treatments) * q0)
    treatment_prediction = treatment_model.predict(xgb.DMatrix(covariates))
    treatment_prediction = np.clip(treatment_prediction, 1e-3, 1 - 1e-3)
    riesz_rep = treatments / treatment_prediction - (1 - treatments) / (1 - treatment_prediction)
    correction = np.mean(riesz_rep * residuals)
    print(plugin, plugin + correction, truth)
    dr_est = plugin + correction
    ests.append(dr_est)
    MAE = sum(np.abs(np.array(truths) - np.array(ests))) / len(truths)
    print("MAE:", MAE)
