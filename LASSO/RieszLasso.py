import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegressionCV, Lasso, LassoCV


class RieszLasso:
    def __init__(self, functional, expand_treatment=False):
        self.functional = functional
        self.rho = None
        self.intercept = None
        self.covariate_indices = None
        self.expand_treatment = expand_treatment

    def set_covariate_indices(self, covariate_indices):
        self.covariate_indices = covariate_indices

    def make_design_matrix(self, data):

        if self.covariate_indices is None:
            covariate_indices = np.arange(data.covariates.shape[1], dtype=np.int32)
        else:
            covariate_indices = self.covariate_indices

        if self.expand_treatment:
            design = np.concatenate(
                [
                    data.treatments.reshape(-1, 1),
                    1 - data.treatments.reshape(-1, 1),
                    data.covariates[:, covariate_indices],
                ],
                axis=1,
            )
        else:
            design = np.concatenate(
                [
                    data.treatments.reshape(-1, 1),
                    data.covariates[:, covariate_indices],
                ],
                axis=1,
            )
        return design

    def fit(self, data, c1=1 / 5, ridge_penalty=0.0):

        xb = self.make_design_matrix(data)
        n, p = xb.shape[0], xb.shape[1]
        mb = self.functional(data, self.make_design_matrix)
        hatM = np.mean(mb, axis=0)
        hatG = 1 / n * (xb.T @ xb)
        hatG_ridge = hatG + np.eye(p) * ridge_penalty
        eigvals, eigvecs = np.linalg.eigh(hatG_ridge)

        eigvals_clipped = np.maximum(eigvals, 1e-10)
        A = np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
        y = np.linalg.solve(A.T, hatM)

        c2 = 0.1
        penalty = c1 / np.sqrt(n) * norm.ppf(1 - c2 / (2 * p))
        penalty_to_sklearn = penalty / (2 * n)

        mod = Lasso(alpha=penalty_to_sklearn, max_iter=5000, fit_intercept=False, selection="random")
        mod.fit(A, y)

        self.rho = mod.coef_
        self.intercept = mod.intercept_

    def get_riesz_loss(self, data):
        mb = self.functional(data, self.make_design_matrix)
        loss = -2 * mb @ self.rho + self.get_riesz_representer(data) ** 2
        return np.mean(loss)

    def fit_cv(self, data, c1s=np.array([5 / 4, 3 / 4, 1 / 2]), n_folds=5):
        if isinstance(c1s, float):
            self.fit(data, c1s)
        else:
            folds = data.split_into_folds(n_folds)
            best_loss = np.inf
            best_c1 = None

            for c1 in c1s:
                cur_loss = 0
                for i in range(n_folds):
                    test_data, train_data = data.get_fit_and_train_folds(folds, i)
                    self.fit(train_data, c1)
                    riesz_loss = self.get_riesz_loss(test_data)
                    cur_loss += riesz_loss

                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_c1 = c1
                print(c1, cur_loss)
            self.fit(data, best_c1)

    def get_riesz_representer(self, data):
        xb = self.make_design_matrix(data)
        return xb @ self.rho + self.intercept


class PropensityLasso:
    def __init__(self):
        self.model = LogisticRegressionCV(
            penalty="l1", fit_intercept=True, solver="liblinear", n_jobs=5, Cs=25, scoring="neg_log_loss", tol=0.001
        )
        self.covariate_indices = None

    def make_design_matrix(self, data):

        if self.covariate_indices is None:
            covariate_indices = np.arange(data.covariates.shape[1], dtype=np.int32)
        else:
            covariate_indices = self.covariate_indices

        return data.covariates[:, covariate_indices]

    def set_covariate_indices(self, covariate_indices):
        self.covariate_indices = covariate_indices

    def fit(self, data):
        covariates = self.make_design_matrix(data)
        treatments = data.treatments
        self.model.fit(covariates, treatments)

    def get_riesz_representer(self, data):
        covariates = self.make_design_matrix(data)
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
