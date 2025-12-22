import numpy as np
from sklearn.linear_model import LassoCV


class OracleLasso:
    def __init__(self, _=None):
        self.model = LassoCV(fit_intercept=True)

    def fit(self, data):
        covariates = data.covariates
        treatments = data.treatments
        self.model.fit(covariates, treatments)

    def get_riesz_representer(self, data):
        covariates = data.covariates
        treatments = data.treatments
        est_mean_value = self.model.predict(covariates)
        est_residual_variance = np.var((est_mean_value - treatments))
        rr = self.get_ratio_conditional_gaussians(treatments, est_mean_value, est_residual_variance)
        return rr

    @staticmethod
    def get_ratio_conditional_gaussians(treatments, mean_value_vector, variance):
        denominator = np.exp(-((treatments - mean_value_vector) ** 2 / (variance * 2)))
        numerator = np.exp(-((treatments - 1 - mean_value_vector) ** 2 / (variance * 2)))
        return numerator / denominator - 1
