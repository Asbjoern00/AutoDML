import numpy as np
from scipy.stats import norm
from LASSO.md_lasso import md_lasso, compute_loadings
from sklearn.linear_model import LogisticRegressionCV, LassoCV


class RieszLasso:
    def __init__(self, functional):
        self.functional = functional
        self.rho = None
        self.covariate_indices = None

    def set_covariate_indices(self, covariate_indices):
        self.covariate_indices = covariate_indices

    def make_design_matrix(self, data, low_dimensional=False):

        if self.covariate_indices is None:
            covariate_indices = np.arange(data.covariates.shape[1], dtype=np.int32)
        else:
            covariate_indices = self.covariate_indices

        if low_dimensional:
            covariate_indices = covariate_indices[: np.ceil(len(covariate_indices) / 40).astype(np.int32)]

        design = np.concatenate(
            [
                data.treatments.reshape(-1, 1),
                data.covariates[:, covariate_indices],
                np.ones(data.treatments.shape[0]).reshape(-1, 1),
            ],
            axis=1,
        )

        return design

    def fit(self, data, c1=0.9, max_it=100, tol = 0.001):

        xb = self.make_design_matrix(data)
        n, p = xb.shape[0], xb.shape[1]
        intercept_indices = np.array([p-1])
        mb = self.functional(data, self.make_design_matrix)
        hatM = np.mean(mb, axis=0)
        hatG = 1 / n * (xb.T @ xb)

        c2 = 0.1
        penalty = c1 / np.sqrt(n) * norm.ppf(1 - c2 / (2 * p))

        if penalty > 0:
            xb_low = self.make_design_matrix(data, low_dimensional=True)
            mb_low = self.functional(data, lambda x: self.make_design_matrix(x, low_dimensional=True))
            hatM_low = np.mean(mb_low, axis=0)
            hatG_low = 1 / n * (xb_low.T @ xb_low)

            rho_init = np.zeros(xb.shape[1])
            rho_init_fit = np.linalg.solve(hatG_low, hatM_low)
            rho_init[: len(rho_init_fit)] = rho_init_fit

            for m in range(max_it):
                hatD = compute_loadings(xb, mb, rho_init, c3=0.1, intercept_indices=intercept_indices)
                rho = md_lasso(hatG, hatM, D=hatD, rL=penalty, rho_init=rho_init, max_iter=10)

                diff = rho_init - rho
                max_change = np.max(np.abs(diff))

                if max_change < tol:
                    break
                else:
                    rho_init = rho
        else:
            rho = np.linalg.solve(hatG, hatM)

        self.rho = rho

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

    def get_riesz_loss(self, data):
        mb = self.functional(data, self.make_design_matrix)
        loss = -2 * mb @ self.rho + self.get_riesz_representer(data) ** 2
        return np.mean(loss)

    def get_riesz_representer(self, data):
        xb = self.make_design_matrix(data)
        return xb @ self.rho


class PropensityLasso:
    def __init__(self):
        self.model = LogisticRegressionCV(
            penalty="l1", fit_intercept=True, solver="liblinear", n_jobs=5, Cs=10, scoring="neg_log_loss", tol = 0.001
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
