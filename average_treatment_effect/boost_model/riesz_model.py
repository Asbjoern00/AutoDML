import numpy as np
import xgboost as xgb


class RieszBooster:
    def __init__(self, params=None):
        self.model = None
        self.params = {"max_depth": 3, "eta": 0.1}
        if params is not None:
            self.params.update(params)

    def fit(self, data, num_boost_round=100):
        data = data.to_riesz_xgb_dataset()["training_data"]
        self.model = xgb.train(
            self.params,
            data,
            num_boost_round=num_boost_round,
            obj=riesz_objective,
        )

    def get_riesz_representer(self, data):
        data = data.to_riesz_xgb_dataset()["test_data"]
        return self.model.predict(data)


def riesz_objective(predictions, data):
    grad = np.zeros_like(predictions)
    hess = np.zeros_like(predictions)
    label = data.get_label()
    grad[label == 2] = 2 * predictions[label == 2]
    grad[label == 0] = 2
    grad[label == 1] = -2
    hess[label == 2] = 2 * np.ones_like(hess[label == 2])
    hess = hess + 1e-3
    return grad, hess


def riesz_eval(predictions, data):
    label = data.get_label()
    loss = predictions[label == 2] ** 2 - 2 * (predictions[label == 1] - predictions[label == 0])
    loss = np.mean(loss)
    return "Riesz-Loss", float(loss)
