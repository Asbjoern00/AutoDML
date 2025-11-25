import numpy as np
import xgboost as xgb


class RieszBooster:
    def __init__(self, params=None, hess=0.1):
        self.model = None
        self.params = {
            "lambda": 33,
            "eta": 0.1,
            "max_depth": 2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "num_parallel_tree": 5,
        }
        if params is not None:
            self.params.update(params)
        self.hess = hess

    def fit(self, data, num_boost_round=100):
        data = data.to_riesz_xgb_dataset()["training_data"]
        self.model = xgb.train(
            self.params,
            data,
            num_boost_round=num_boost_round,
            obj=self.riesz_objective,
        )

    def cross_validate(self, data, num_boost_round=100):
        data = data.to_riesz_xgb_dataset()["training_data"]
        cv = xgb.cv(
            self.params,
            data,
            num_boost_round=num_boost_round,
            obj=self.riesz_objective,
            custom_metric=riesz_eval,
            nfold=5,
            stratified=True,
            shuffle=True,
        )
        test_error = np.asarray(cv["test-Riesz-Loss-mean"], dtype=float)
        train_error = np.asarray(cv["train-Riesz-Loss-mean"], dtype=float)
        return {
            "best_test_round": int(np.nanargmin(test_error)),
            "best_test_error": float(test_error[int(np.nanargmin(test_error))]),
            "best_train_round": int(np.nanargmin(train_error)),
            "best_train_error": float(train_error[int(np.nanargmin(train_error))]),
        }

    def get_riesz_representer(self, data):
        data = data.to_riesz_xgb_dataset()["test_data"]
        return self.model.predict(data)

    def riesz_objective(self, predictions, data):
        grad = np.zeros_like(predictions)
        hess = np.zeros_like(predictions)
        label = data.get_label()
        grad[label == 2] = 2 * predictions[label == 2]
        grad[label == 0] = 2
        grad[label == 1] = -2
        hess[label == 2] = 2 * np.ones_like(hess[label == 2])
        hess = hess + self.hess
        return grad, hess


def riesz_eval(predictions, data):
    label = data.get_label()
    loss = np.mean(predictions[label == 2] ** 2) - 2 * (
        np.mean(predictions[label == 1]) - np.mean(predictions[label == 0])
    )
    return "Riesz-Loss", float(loss)
