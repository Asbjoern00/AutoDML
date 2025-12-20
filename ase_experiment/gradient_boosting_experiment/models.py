import numpy as np
import xgboost as xgb

from ase_experiment.dataset import Dataset


class OutcomeXGBModel:
    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, data: Dataset):
        data_train, data_test = data.test_train_split(train_proportion=0.8)
        self.model = xgb.train(
            params=self.params,
            dtrain=data_train.xgb_dataset,
            num_boost_round=1000,
            evals=[(data_train.xgb_dataset, "train"), (data_test.xgb_dataset, "eval")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

    def get_predictions(self, data):
        treated_data, control_data = data.get_counterfactual_datasets()
        predictions = self.model.predict(data.xgb_dataset)
        treated_predictions = self.model.predict(treated_data.xgb_dataset)
        control_predictions = self.model.predict(control_data.xgb_dataset)
        return {
            "predictions": predictions,
            "treated_predictions": treated_predictions,
            "control_predictions": control_predictions,
        }


class TreatmentXGBModel:
    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, data: Dataset):
        data_train, data_test = data.test_train_split(train_proportion=0.8)
        self.model = xgb.train(
            params=self.params,
            dtrain=data_train.xgb_treatment_dataset,
            num_boost_round=1000,
            evals=[(data_train.xgb_treatment_dataset, "train"), (data_test.xgb_treatment_dataset, "eval")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

    def get_riesz_representer(self, data):
        predictions = self.model.predict(data.xgb_treatment_dataset)
        riesz_representer = (
            np.exp(-1 / 4 * (data.treatments - 1 - predictions) ** 2 + 1 / 4 * (data.treatments - predictions) ** 2) - 1
        )
        return riesz_representer


class RieszXGBModel:
    def __init__(self, params, hessian_correction=0):
        self.model = None
        self.params = params
        self.hessian_correction = hessian_correction

    def fit(self, data: Dataset):
        data_train, data_test = data.test_train_split(train_proportion=0.8)
        self.model = xgb.train(
            params=self.params,
            dtrain=data_train.xgb_riesz_dataset,
            num_boost_round=1000,
            obj=self.riesz_objective,
            custom_metric=self.riesz_eval,
            evals=[(data_train.xgb_riesz_dataset, "train"), (data_test.xgb_riesz_dataset, "eval")],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

    def get_riesz_representer(self, data):
        propensity_scores = self.model.predict(data.xgb_riesz_dataset)
        riesz_representer = propensity_scores[: data.raw_data.shape[0]]
        return riesz_representer

    def riesz_objective(self, predictions, data):
        grad = np.zeros_like(predictions)
        hess = np.zeros_like(predictions)
        label = data.get_label()
        grad[label == 2] = 2 * predictions[label == 2]
        grad[label == 0] = 2
        grad[label == 1] = -2
        hess[label == 2] = 2 * np.ones_like(hess[label == 2])
        hess = hess + self.hessian_correction
        return grad, hess

    @staticmethod
    def riesz_eval(predictions, data):
        label = data.get_label()
        loss = np.mean(predictions[label == 2] ** 2) - 2 * (
            np.mean(predictions[label == 1]) - np.mean(predictions[label == 0])
        )
        return "Riesz-Loss", float(loss)
