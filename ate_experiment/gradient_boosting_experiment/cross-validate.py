import numpy as np

from ate_experiment.dataset import Dataset
from ate_experiment.gradient_boosting_experiment.models import OutcomeXGBModel, PropensityXGBModel, RieszXGBModel

np.random.seed(20122025)


depths = [2, 3, 4, 5]
etas = [0.01, 0.05, 0.1, 0.3]
lambdas = [1, 10, 50, 100]
data = Dataset.simulate_dataset(1000, 10)


best = 1e3
for depth in depths:
    for eta in etas:
        for lambda_ in lambdas:
            params = {
                "disable_default_eval_metric": True,
                "max_depth": depth,
                "eta": eta,
                "lambda": lambda_,
            }
            model = RieszXGBModel(params)
            res = model.cv(data)
            if res < best:
                best = res
                best_params = params
print(best_params)

best = 1e3
for depth in depths:
    for eta in etas:
        for lambda_ in lambdas:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "eta": eta,
                "lambda": lambda_,
                "max_depth": depth,
            }
            model = OutcomeXGBModel(params)
            res = model.cv(data)
            if res < best:
                best = res
                best_params = params
print(best_params)


best = 1e3
for depth in depths:
    for eta in etas:
        for lambda_ in lambdas:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "eta": eta,
                "lambda": lambda_,
                "max_depth": depth,
            }
            model = PropensityXGBModel(params)
            res = model.cv(data)
            if res < best:
                best = res
                best_params = params
print(best_params)
