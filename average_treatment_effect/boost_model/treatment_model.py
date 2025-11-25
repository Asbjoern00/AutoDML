import numpy as np
import xgboost as xgb


class TreatmentBooster:
    def __init__(self, params=None):
        self.model = None
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.1,
            "max_depth": 2,
            "lambda": 20,
        }
        if not params is None:
            self.params.update(params)

    def fit(self, data, num_boost_round=300):
        data = data.to_xgb_dataset()["treatment_dataset"]
        self.model = xgb.train(self.params, data, num_boost_round=num_boost_round)

    def cross_validate(self, data, num_boost_round=300):
        data = data.to_xgb_dataset()["treatment_dataset"]
        cv0 = xgb.cv(self.params, data, num_boost_round=num_boost_round, nfold=5, stratified=True, shuffle=True)
        rmse = np.asarray(cv0["test-logloss-mean"], dtype=float)
        return {
            "best_round": int(np.nanargmin(rmse)),
            "best_error": float(rmse[int(np.nanargmin(rmse))]),
        }

    def get_riesz_representer(self, data):
        treatments = data.treatments[:, 0]
        data = data.to_xgb_dataset()["full_covariates"]
        predictions = self.model.predict(data)
        return treatments / predictions - (1 - treatments) / (1 - predictions)
