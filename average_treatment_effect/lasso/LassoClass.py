import numpy as np
from average_treatment_effect.lasso.OutcomeLASSO import OutcomeLASSO
from average_treatment_effect.lasso.RieszLasso import RieszLasso
from average_treatment_effect.Functional.ATEFunctional import ate_functional


class LassoATE:
    def __init__(self, riesz_model):
        self.riesz_model = riesz_model(ate_functional)
        self.outcome_model = OutcomeLASSO(ate_functional)

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
