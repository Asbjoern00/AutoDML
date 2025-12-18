import numpy as np

from ate_experiment.dataset import Dataset
from ate_experiment.gradient_boosting_experiment.models import OutcomeXGBModel

data = Dataset.simulate_dataset(number_of_samples=1000, number_of_covariates=5)
params = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 3, "eta": 0.1, "lambda": 3}
model = OutcomeXGBModel(params)
model.fit(data)
outcome_predictions = model.get_predictions(data)
plug_in = np.mean(outcome_predictions["treated_predictions"] - outcome_predictions["control_predictions"])
print(plug_in)
