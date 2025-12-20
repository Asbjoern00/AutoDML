import numpy as np
from average_treatment_effect.lasso.md_lasso import md_lasso


class RieszLasso:
    def __init__(self, functional, penalty = 0.05):
        self.functional = functional
        self.penalty = penalty
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

    def fit(self, data):
        xb = self.make_design_matrix(data)
        mb = self.functional(data, self.make_design_matrix)
        hatM = np.mean(mb, axis=0)
        hatG = 1 / xb.shape[0] * (xb.T @ xb)
        rho = md_lasso(hatG, hatM, penalty=self.penalty)
        self.rho = rho

    def get_riesz_representer(self, data):
        xb = self.make_design_matrix(data)
        return xb @ self.rho
