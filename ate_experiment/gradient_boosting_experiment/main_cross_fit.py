import numpy as np

from ate_experiment.dataset import Dataset
from ate_experiment.gradient_boosting_experiment.models import OutcomeXGBModel, PropensityXGBModel, RieszXGBModel

np.random.seed(2131)

truth = 2.121539888279284
plug_ins = []
propensity_ests = []
riesz_ests = []
propensity_vars = []
riesz_vars = []
riesz_covers = []
propensity_covers = []

iterations = 1000
number_of_samples = 1000
number_of_covariates = 10
lambda_ = 5
number_of_folds = 10

outcome_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 3,
    "eta": 0.1,
    "lambda": lambda_,
}

propensity_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1,
    "lambda": lambda_,
}

riesz_params = {"disable_default_eval_metric": True, "max_depth": 2, "eta": 0.1, "lambda": 30}


for i in range(iterations):

    print(i)

    data = Dataset.simulate_dataset(number_of_samples=number_of_samples, number_of_covariates=number_of_covariates)
    folds = data.split_into_folds(number_of_folds)

    plug_in_terms = np.array([])
    propensity_terms = np.array([])
    riesz_terms = np.array([])

    for j in range(number_of_folds):
        fit_fold, train_folds = Dataset.get_fit_and_train_folds(folds, j)
        outcome_model = OutcomeXGBModel(outcome_params)
        outcome_model.fit(train_folds)
        propensity_model = PropensityXGBModel(propensity_params)
        propensity_model.fit(train_folds)
        riesz_model = RieszXGBModel(riesz_params, hessian_correction=0)
        riesz_model.fit(train_folds)

        outcome_predictions = outcome_model.get_predictions(fit_fold)
        residuals = fit_fold.outcomes - outcome_predictions["predictions"]
        propensity_riesz_representer = propensity_model.get_riesz_representer(fit_fold)
        riesz_riesz_representer = riesz_model.get_riesz_representer(fit_fold)

        plug_in_term = outcome_predictions["treated_predictions"] - outcome_predictions["control_predictions"]
        propensity_term = plug_in_term + propensity_riesz_representer * residuals
        riesz_term = plug_in_term + riesz_riesz_representer * residuals

        plug_in_terms = np.concatenate([plug_in_terms, plug_in_term])
        propensity_terms = np.concatenate([propensity_terms, propensity_term])
        riesz_terms = np.concatenate([riesz_terms, riesz_term])

    plug_in = np.sum(plug_in_terms) / number_of_samples

    propensity_estimate = np.sum(propensity_terms) / number_of_samples
    propensity_var = np.sum((propensity_terms - propensity_estimate) ** 2) / (number_of_samples**2)
    propensity_lower = propensity_estimate - 1.96 * np.sqrt(propensity_var)
    propensity_upper = propensity_estimate + 1.96 * np.sqrt(propensity_var)
    propensity_cover = propensity_lower <= truth <= propensity_upper


    riesz_estimate = np.sum(riesz_terms) / number_of_samples
    riesz_var = np.sum((riesz_terms - riesz_estimate) ** 2) / (number_of_samples**2)
    riesz_lower = riesz_estimate - 1.96 * np.sqrt(riesz_var)
    riesz_upper = riesz_estimate + 1.96 * np.sqrt(riesz_var)
    riesz_cover = riesz_lower <= truth <= riesz_upper

    plug_ins.append(plug_in)
    riesz_ests.append(riesz_estimate)
    propensity_ests.append(propensity_estimate)
    riesz_covers.append(riesz_cover)
    propensity_covers.append(propensity_cover)

    plugin_mse = sum((est - truth) ** 2 for est in plug_ins) / len(plug_ins)
    propensity_mse = sum((est - truth) ** 2 for est in propensity_ests) / len(propensity_ests)
    riesz_mse = sum((est - truth) ** 2 for est in riesz_ests) / len(riesz_ests)
    propensity_coverage = sum(propensity_covers) / len(propensity_covers)
    riesz_coverage = sum(riesz_covers) / len(riesz_covers)


    print(plugin_mse**0.5, propensity_mse**0.5, riesz_mse**0.5)
    print(propensity_coverage, riesz_coverage)
