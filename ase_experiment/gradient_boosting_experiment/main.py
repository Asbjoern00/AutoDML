import numpy as np

from ase_experiment.dataset import Dataset
from ase_experiment.gradient_boosting_experiment.models import OutcomeXGBModel, RieszXGBModel

np.random.seed(2131)

truth = 108.997


plug_ins = []
propensity_ests = []
riesz_ests = []
riesz_covers = []
propensity_covers = []
riesz_lowers = []
riesz_uppers = []
propensity_lowers = []
propensity_uppers = []
riesz_vars = []
propensity_vars = []

iterations = 1000
number_of_samples = 1000
number_of_covariates = 1
number_of_folds = 10

outcome_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 3,
    "eta": 0.1,
}

riesz_params = {
    "disable_default_eval_metric": True,
    "max_depth": 2,
    "eta": 0.1,
}


for i in range(iterations):

    print(i)

    data = Dataset.simulate_dataset(number_of_samples=number_of_samples, number_of_covariates=number_of_covariates)
    outcome_model = OutcomeXGBModel(outcome_params)
    outcome_model.fit(data)
    outcome_predictions = outcome_model.get_predictions(data)
    plug_in = np.mean(outcome_predictions["treated_predictions"] - outcome_predictions["control_predictions"])

    riesz_model = RieszXGBModel(riesz_params, hessian_correction=0)
    riesz_model.fit(data)
    riesz_riesz_representer = riesz_model.get_riesz_representer(data)
    riesz_estimate_terms = (
        outcome_predictions["treated_predictions"]
        - outcome_predictions["control_predictions"]
        + riesz_riesz_representer * (data.outcomes - outcome_predictions["predictions"])
    )
    riesz_estimate = np.sum(riesz_estimate_terms) / number_of_samples
    riesz_var = np.sum((riesz_estimate_terms - riesz_estimate) ** 2) / (number_of_samples**2)
    riesz_lower = riesz_estimate - 1.96 * np.sqrt(riesz_var)
    riesz_upper = riesz_estimate + 1.96 * np.sqrt(riesz_var)
    riesz_cover = riesz_lower <= truth <= riesz_upper

    plug_ins.append(plug_in)
    riesz_ests.append(riesz_estimate)
    riesz_covers.append(riesz_cover)
    riesz_uppers.append(riesz_upper)
    riesz_lowers.append(riesz_lower)
    riesz_vars.append(riesz_var)

    plugin_mse = sum((est - truth) ** 2 for est in plug_ins) / len(plug_ins)
    riesz_mse = sum((est - truth) ** 2 for est in riesz_ests) / len(riesz_ests)
    riesz_coverage = sum(riesz_covers) / len(riesz_covers)

    print(plugin_mse**0.5, riesz_mse**0.5)
    print(riesz_coverage)
