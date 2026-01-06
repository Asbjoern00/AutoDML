import numpy as np
from LASSO.md_lasso import md_lasso
from sklearn.linear_model import LogisticRegressionCV, LassoCV


class RieszLasso:
    def __init__(self, functional):
        self.functional = functional
        self.rho = None
        self.covariate_indices = None

    def set_covariate_indices(self, covariate_indices):
        self.covariate_indices = covariate_indices

    def make_design_matrix(self,data):
        if self.covariate_indices is None:
            covariate_indices = np.arange(data.covariates.shape[1], dtype=np.int32)
        else:
            covariate_indices = self.covariate_indices

        design = np.concatenate(
                [
                    data.treatments.reshape(-1, 1),
                    data.covariates[:, covariate_indices],
                    np.ones(data.treatments.shape[0]).reshape(-1, 1)
                ],
                axis=1,
            )

        return design

    def fit(self, data, penalty=0.05):
        xb = self.make_design_matrix(data)
        mb = self.functional(data,self.make_design_matrix)
        hatM = np.mean(mb, axis=0)
        hatG = 1 / xb.shape[0] * (xb.T @ xb)

        if penalty > 0:
            rho = md_lasso(hatG, hatM, rL=penalty, rho_init=self.rho)  # If rho is already fitted, warm state from rho
        else:
            rho = np.linalg.solve(hatG, hatM)
        self.rho = rho

    def fit_cv(self, data, penalties=np.array([0.2, 0.15, 0.1, 0.075, 0.05]), n_folds=5):
        folds = data.split_into_folds(n_folds)
        best_loss = np.inf
        best_rho = None

        for penalty in penalties:
            cur_loss = 0
            for i in range(n_folds):
                test_data, train_data = data.get_fit_and_train_folds(folds, i)
                self.fit(train_data, penalty)
                riesz_loss = self.get_riesz_loss(test_data)
                cur_loss += riesz_loss

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_rho = penalty

        self.fit(data, best_rho)

    def get_riesz_loss(self, data):
        mb = self.functional(data, self.make_design_matrix)
        loss = -2 * mb @ self.rho + self.get_riesz_representer(data) ** 2
        return np.mean(loss)

    def get_riesz_representer(self, data):
        xb = self.make_design_matrix(data)
        return xb @ self.rho


class PropensityLasso:
    def __init__(self):
        self.model = LogisticRegressionCV(penalty="l1", fit_intercept=True, solver="liblinear", n_jobs=5, Cs=10)

    def fit(self, data):
        covariates = data.covariates
        treatments = data.treatments
        self.model.fit(covariates, treatments)

    def get_riesz_representer(self, data):
        covariates = data.covariates
        treatments = data.treatments
        pi = self.model.predict_proba(covariates)[:, self.model.classes_ == 1].reshape(
            treatments.shape[0],
        )

        clip_lower = 1 / 100
        clip_upper = 1 - 1 / 100
        pi = np.clip(pi, clip_lower, clip_upper)

        return treatments / pi - (1 - treatments) / (1 - pi)


class ASETreatmentLasso:
    def __init__(self):
        self.model = LassoCV(fit_intercept=True, n_jobs=5)

    def fit(self, data):
        covariates = data.covariates
        treatments = data.treatments
        self.model.fit(covariates, treatments)

    @staticmethod
    def _gaussian_density_ratio(u, mu, variance=1, shift=1):
        return np.exp(-1 / (2 * variance) * (u - shift - mu) ** 2) / np.exp(-1 / (2 * variance) * (u - mu) ** 2) - 1

    def get_riesz_representer(self, data):
        covariates = data.covariates
        treatments = data.treatments
        mean_outcome = self.model.predict(covariates)
        density_ratio = self._gaussian_density_ratio(treatments, mean_outcome)

        clip_lower = -100
        clip_upper = 100
        rr = np.clip(density_ratio, clip_lower, clip_upper)

        return rr
