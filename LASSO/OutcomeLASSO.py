import numpy as np
from sklearn.linear_model import LassoCV


class OutcomeLASSO:
    def __init__(self, functional):
        self.model = LassoCV(fit_intercept=True, n_jobs=5)
        self.functional = functional

    def fit(self, data):
        X = np.concatenate([data.treatments.reshape(-1, 1), data.covariates], axis=1)
        y = data.outcomes
        self.model.fit(X, y)

    def predict(self, data):
        X = np.concatenate([data.treatments.reshape(-1, 1), data.covariates], axis=1)
        return self.model.predict(X)

    def get_residuals(self, data):
        fitted = self.predict(data)
        observed = data.outcomes
        return observed - fitted

    def get_functional(self, data):
        return self.functional(data, self.predict)

    def get_plugin_estimate(self, data):
        return np.mean(self.get_functional(data))

    def get_active_covariate_indices(self):
        coef = self.model.coef_
        non_zero_indices = np.nonzero(coef)[0]
        non_zero_covariate_indices = non_zero_indices[non_zero_indices != 0] # treatment is located at index 0. exclude
        non_zero_covariate_indices = non_zero_covariate_indices - np.array(1) # treatment is located at index 0, shift rest of array

        return non_zero_covariate_indices.astype(np.int32)
