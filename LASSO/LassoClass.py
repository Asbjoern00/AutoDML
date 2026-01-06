import numpy as np
from LASSO.RieszLasso import RieszLasso


class Lasso:
    def __init__(self, riesz_model, outcome_model):
        self.riesz_model = riesz_model
        self.outcome_model = outcome_model

    def fit(self, data, cv_riesz=True):
        self.outcome_model.fit(data)
        if isinstance(self.riesz_model, RieszLasso) and cv_riesz:
            self.riesz_model.fit_cv(data)
        else:
            self.riesz_model.fit(data)

    def get_plugin(self, data):
        return self.outcome_model.get_plugin_estimate(data)

    def get_correction(self, data):
        residuals = self.outcome_model.get_residuals(data)
        rr = self.riesz_model.get_riesz_representer(data)
        return residuals * rr

    def get_functional(self, data):
        return self.outcome_model.get_functional(data)

    def get_double_robust(self, data):
        plugin = self.get_plugin(data)
        correction = self.get_correction(data)
        return plugin + np.mean(correction)

class OutcomeAdaptedLasso(Lasso):
    def __init__(self, riesz_model, outcome_model):
        super().__init__(riesz_model, outcome_model)

    def fit(self, data,cv_riesz=None):
        self.outcome_model.fit(data)
        active_covariate_indices = self.outcome_model.get_active_covariate_indices()
        print(len(active_covariate_indices))
        self.riesz_model.set_covariate_indices(active_covariate_indices)
        self.riesz_model.fit_cv(data, penalties=np.array([0.2, 0.15, 0.1, 0.075, 0.05,0]))
