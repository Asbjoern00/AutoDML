import numpy as np
import xgboost as xgb

from ate_experiment.dataset import Dataset


class OutcomeXGBModel:
    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, data: Dataset):
        data_train, data_test = data.test_train_split(train_proportion=0.8)
        self.model = xgb.train(
            params=self.params,
            dtrain=data_train.xgb_dataset,
            num_boost_round=10000,
            evals=[(data_train.xgb_dataset, "train"), (data_test.xgb_dataset, "eval")],
            early_stopping_rounds=20,
            verbose_eval=True,
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


class PropensityXGBModel:
    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, data: Dataset):
        data_train, data_test = data.test_train_split(train_proportion=0.8)
        self.model = xgb.train(
            params=self.params,
            dtrain=data_train.xgb_propensity_dataset,
            num_boost_round=10000,
            evals=[(data_train.xgb_propensity_dataset, "train"), (data_test.xgb_propensity_dataset, "eval")],
            early_stopping_rounds=20,
            verbose_eval=True,
        )

    def get_riesz_representer(self, data):
        propensity_scores = self.model.predict(data.xgb_propensity_dataset)
        return data.treatments / propensity_scores + (1 - data.treatments) / (1 - propensity_scores)
