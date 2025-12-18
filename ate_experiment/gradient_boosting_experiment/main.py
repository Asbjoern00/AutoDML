import numpy as np

from ate_experiment.dataset import Dataset
from ate_experiment.gradient_boosting_experiment.models import OutcomeXGBModel, PropensityXGBModel

data = Dataset.simulate_dataset(number_of_samples=1000, number_of_covariates=5)
outcome_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 3, "eta": 0.1}
outcome_model = OutcomeXGBModel(outcome_params)
outcome_model.fit(data)
outcome_predictions = outcome_model.get_predictions(data)
plug_in = np.mean(outcome_predictions["treated_predictions"] - outcome_predictions["control_predictions"])
propensity_params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.1}
propensity_model = PropensityXGBModel(propensity_params)
propensity_model.fit(data)
riesz_representer = propensity_model.get_riesz_representer(data)
correction = np.mean(riesz_representer * (data.outcomes - outcome_predictions["predictions"]))
print(riesz_representer)
print(plug_in)
print(correction)
print(plug_in + correction)
