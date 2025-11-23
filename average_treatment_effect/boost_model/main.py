import xgboost as xgb
import numpy as np

from average_treatment_effect.dataset import Dataset

data = Dataset.load_chernozhukov_replication(300)
treatments = data.treatments
outcomes = data.outcomes
covariates = data.covariates

outcome_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "eta": 0.1, "max_depth": 5}
outcome_model = xgb.train(
    outcome_params,
    xgb.DMatrix(np.column_stack([treatments, covariates]), label=outcomes),
    num_boost_round=100,
    evals=[((xgb.DMatrix(np.column_stack([treatments, covariates]), label=outcomes)), "train")],
)
q1 = outcome_model.predict(xgb.DMatrix(np.column_stack([np.ones_like(treatments), covariates]), label=outcomes))
q0 = outcome_model.predict(xgb.DMatrix(np.column_stack([np.zeros_like(treatments), covariates]), label=outcomes))
plugin = np.mean(q1 - q0)

treatment_params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.1, "max_depth": 5}
treatment_model = xgb.train(
    treatment_params,
    xgb.DMatrix(covariates, label=treatments),
    num_boost_round=100,
    evals=[((xgb.DMatrix(covariates, label=treatments)), "train")],
)
residuals = outcomes[:, 0] - outcome_model.predict(xgb.DMatrix(np.column_stack([treatments, covariates])))
treatment_prediction = treatment_model.predict(xgb.DMatrix(covariates))
riesz_rep = treatments[:, 0] / treatment_prediction - (1 - treatments[:, 0]) / (1 - treatment_prediction)
correction = np.mean(riesz_rep * residuals)

print(data.get_average_treatment_effect())
print(plugin)
print(plugin + correction)
