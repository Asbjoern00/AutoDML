import numpy as np

from ate_experiment.dataset import Dataset
from ate_experiment.gradient_boosting_experiment.models import OutcomeXGBModel, PropensityXGBModel, RieszXGBModel

truth = 29.502
plug_ins = []
propensity_ests = []
riesz_ests = []

iterations = 1000
number_of_samples = 1000
lambda_ = 3


for i in range(iterations):

    print(i)

    data = Dataset.simulate_dataset(number_of_samples=number_of_samples, number_of_covariates=1)
    outcome_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 3,
        "eta": 0.1,
        "lambda": lambda_,
    }
    outcome_model = OutcomeXGBModel(outcome_params)
    outcome_model.fit(data)
    outcome_predictions = outcome_model.get_predictions(data)
    plug_in = np.mean(outcome_predictions["treated_predictions"] - outcome_predictions["control_predictions"])

    propensity_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.1,
        "lambda": lambda_,
    }
    propensity_model = PropensityXGBModel(propensity_params)
    propensity_model.fit(data)
    propensity_riesz_representer = propensity_model.get_riesz_representer(data)
    propensity_correction = np.mean(propensity_riesz_representer * (data.outcomes - outcome_predictions["predictions"]))
    propensity_estimate = plug_in + propensity_correction

    riesz_params = {"disable_default_eval_metric": True, "max_depth": 3, "eta": 0.1, "lambda": lambda_}
    riesz_model = RieszXGBModel(riesz_params, hessian_correction=0)
    riesz_model.fit(data)
    riesz_riesz_representer = riesz_model.get_riesz_representer(data)
    riesz_correction = np.mean(riesz_riesz_representer * (data.outcomes - outcome_predictions["predictions"]))
    riesz_estimate = plug_in + riesz_correction

    plug_ins.append(plug_in)
    riesz_ests.append(riesz_estimate)
    propensity_ests.append(propensity_estimate)

plugin_mse = sum((est - truth) ** 2 for est in plug_ins) / len(plug_ins)
propensity_mse = sum((est - truth) ** 2 for est in propensity_ests) / len(propensity_ests)
riesz_mse = sum((est - truth) ** 2 for est in riesz_ests) / len(riesz_ests)

print(plugin_mse, propensity_mse, riesz_mse)
