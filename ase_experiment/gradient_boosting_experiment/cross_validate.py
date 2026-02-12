import numpy as np

from ase_experiment.dataset import Dataset
from ase_experiment.gradient_boosting_experiment.models import OutcomeXGBModel, TreatmentXGBModel, RieszXGBModel

depths = [2, 3, 4, 5]
etas = [0.01, 0.05, 0.1, 0.3]
lambdas = [1, 10, 50, 100]

def cross_validate(data):
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
    riesz_params = best_params

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
    reg_params=best_params

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
                model = TreatmentXGBModel(params)
                res = model.cv(data)
                if res < best:
                    best = res
                    best_params = params
    prop_params=best_params

    return reg_params, riesz_params, prop_params
