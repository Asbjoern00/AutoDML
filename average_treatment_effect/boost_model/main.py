import xgboost as xgb
import numpy as np

from average_treatment_effect.dataset import Dataset
from average_treatment_effect.boost_model.outcome_model import OutcomeBooster

ests = []
truths = []

for i in range(1000):
    data = Dataset.load_chernozhukov_replication(i + 1)
    truth = data.get_average_treatment_effect()
    truths.append(truth)
    model = OutcomeBooster()
    model.fit(data, num_boost_round=1000)
    plugin = model.get_plugin_estimate(data)
    residuals = model.get_residuals(data)

    treatments = data.treatments[:, 0]
    outcomes = data.outcomes[:, 0]
    covariates = data.covariates
    treatment_params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.1, "max_depth": 5}
    treatment_model = xgb.train(
        treatment_params,
        xgb.DMatrix(covariates, label=treatments),
        num_boost_round=100,
    )
    treatment_prediction = treatment_model.predict(xgb.DMatrix(covariates))
    treatment_prediction = np.clip(treatment_prediction, 1e-3, 1 - 1e-3)
    riesz_rep = treatments / treatment_prediction - (1 - treatments) / (1 - treatment_prediction)
    correction = np.mean(riesz_rep * residuals)
    print(plugin, plugin + correction, truth)
    dr_est = plugin + correction
    ests.append(dr_est)
    MAE = sum(np.abs(np.array(truths) - np.array(ests))) / len(truths)
    print("MAE:", MAE)
