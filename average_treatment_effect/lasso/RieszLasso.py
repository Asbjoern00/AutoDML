import numpy as np
from average_treatment_effect.lasso.md_lasso import md_lasso
from sklearn.linear_model import LogisticRegressionCV

class RieszLasso:
    def __init__(self, functional):
        self.functional = functional
        self.rho = None

    @staticmethod
    def make_design_matrix(data, include_intercept=True):
        if include_intercept:
            design = np.concatenate(
            [np.ones(data.treatments.shape[0]).reshape(-1,1), data.treatments.reshape(-1, 1), data.covariates], axis=1
        )
        else:
            design = np.concatenate([data.treatments.reshape(-1, 1), data.covariates], axis=1)

        return design

    def fit(self, data, penalty = 0.05):
        xb = self.make_design_matrix(data)
        mb = self.functional(data, self.make_design_matrix)
        hatM = np.mean(mb, axis=0)
        hatG = 1 / xb.shape[0] * (xb.T @ xb)
        #If rho is already fitted, warm state from rho
        rho = md_lasso(hatG, hatM, penalty=penalty, rho_init=self.rho)
        self.rho = rho

    def get_riesz_representer(self, data):
        xb = self.make_design_matrix(data)
        return xb @ self.rho


class PropensityLasso:
    def __init__(self):
        self.model = LogisticRegressionCV(penalty="l1", fit_intercept=True, solver = "liblinear")

    def fit(self, data):
        covariates = data.covariates
        treatments = data.treatments
        self.model.fit(covariates, treatments)

    def get_riesz_representer(self, data):
        covariates = data.covariates
        treatments = data.treatments
        pi = self.model.predict_proba(covariates)[:,self.model.classes_ == 1]
        return treatments/pi - (1-treatments)/(1-pi)