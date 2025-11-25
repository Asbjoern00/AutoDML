import numpy as np
import xgboost as xgb


class OutcomeBooster:
    def __init__(self, params0=None, params1=None):
        self.model0 = None
        self.model1 = None

        self.params0 = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 4, "eta": 0.05}
        self.params1 = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 4, "eta": 0.05}
        if not params0 is None:
            self.params0.update(params0)
        if not params1 is None:
            self.params1.update(params1)

    def fit(self, data, num_boost_round=300):
        datasets = data.to_xgb_dataset()
        data0 = datasets["outcome_dataset_0"]
        data1 = datasets["outcome_dataset_1"]

        self.model0 = xgb.train(self.params0, data0, num_boost_round=num_boost_round)
        self.model1 = xgb.train(self.params1, data1, num_boost_round=num_boost_round)

    def cross_validate(self, data, num_boost_round=300):
        datasets = data.to_xgb_dataset()
        data0 = datasets["outcome_dataset_0"]
        data1 = datasets["outcome_dataset_1"]
        cv0 = xgb.cv(self.params0, data0, num_boost_round=num_boost_round, nfold=8, shuffle=True)
        cv1 = xgb.cv(self.params1, data1, num_boost_round=num_boost_round, nfold=8, shuffle=True)
        rmse0 = np.asarray(cv0["test-rmse-mean"], dtype=float)
        rmse1 = np.asarray(cv1["test-rmse-mean"], dtype=float)
        return {
            "best_round_0": int(np.nanargmin(rmse0)),
            "best_error_0": float(rmse0[int(np.nanargmin(rmse0))]),
            "best_round_1": int(np.nanargmin(rmse1)),
            "best_error_1": float(rmse1[int(np.nanargmin(rmse1))]),
        }

    def get_plugin_estimate(self, data):
        data = data.to_xgb_dataset()["full_covariates"]
        Y0 = self.model0.predict(data)
        Y1 = self.model1.predict(data)
        return np.mean(Y1 - Y0)

    def get_residuals(self, data):
        treatments = data.treatments[:, 0]
        outcomes = data.outcomes[:, 0]
        data = data.to_xgb_dataset()["full_covariates"]
        Y0 = self.model0.predict(data)
        Y1 = self.model1.predict(data)
        Y = treatments * Y1 + (1 - treatments) * Y0
        return outcomes - Y
