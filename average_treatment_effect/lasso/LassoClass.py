from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def md_lasso(G, M, rL, D=None, rho_init=None, max_iter=5000, tol=1e-3):
    """
    Minimum-distance LASSO by coordinate descent implementing the
    update from the paper (objective scaled by 1/2):

        min_rho  (1/2) rho' G rho - rho' M + rL * || D rho ||_1

    Inputs
    ------
    G : (p,p) ndarray
        Symmetric Gram-like matrix (may be singular / not PD).
    M : (p,) ndarray
        Vector M (paper's notation).
    rL : float
        Penalty multiplier (paper's r_L).
    D : (p,) ndarray or None
        Diagonal loadings; if None, set to ones.
    rho_init : (p,) ndarray or None
        Initial guess for rho; if None use zeros.
    max_iter : int
        Max number of coordinate sweeps.
    tol : float
        Convergence tolerance on L2 change of rho.
    diag_eps : float
        Small positive number used if G[j,j] is (near) zero for stability.
    verbose : bool
        If True print progress.

    Returns
    -------
    rho : (p,) ndarray
        Estimated coefficients.
    info : dict
        Diagnostics: {'n_iter': int, 'converged': bool, 'final_obj': float}
    """
    p = M.shape[0]

    if D is None:
        D = np.ones(p, dtype=float)
    else:
        D = np.asarray(D, dtype=float)

    if rho_init is None:
        rho = np.zeros(p, dtype=float)
    else:
        rho = np.array(rho_init, dtype=float)

    z = np.diag(G).copy()

    for it in range(1, max_iter + 1):
        rho_old = rho.copy()
        change = 0.0
        for j in range(p):
            Grho = G @ rho
            pi = M - (Grho - z * rho)
            pi_j = pi[j]

            if j == p - 1:
                new_rj = pi_j / z[j]  # no penalization consider if this is the right thing to do

            else:
                thresh = D[j] * rL

                if pi_j < -thresh:
                    new_rj = (pi_j + thresh) / z[j]
                elif pi_j > thresh:
                    new_rj = (pi_j - thresh) / z[j]
                else:
                    new_rj = 0.0

            delta = new_rj - rho[j]
            if delta != 0.0:
                change = np.max([change,np.abs(rho_old[j] - new_rj)])
                rho[j] = new_rj
        #print(it,change*np.max(rho))

        if change*np.max(rho) < tol: #Sklearn convergence metrix
            converged = True
            break
    else:
        converged = False

    return rho


class CovariateExpander:
    def __init__(self, spline_degree=3, monomial_degree=2):
        self.spline_degree = spline_degree
        self.poly = PolynomialFeatures(degree=monomial_degree, include_bias=False, interaction_only=True)
        self.continuous_indices = None
        self.passthrough_indices = None
        self.preprocessor = None
        self.purge_cols = None

    @staticmethod
    def _is_numeric_binary(col):
        unique = np.unique(col[~np.isnan(col)])
        return len(unique) <= 2 and np.all(np.isclose(unique, unique.astype(int)))

    def fit(self, x):
        x = np.asarray(x)

        # ---- Step 1: Polynomial expansion ----
        x_poly = self.poly.fit_transform(x)
        self.purge_cols = np.isclose(np.std(x_poly, axis=0), 0)  # Remove columns without variability
        x_poly = x_poly[:, ~self.purge_cols]

        # ---- Step 2: Identify column types ----
        continuous = []
        passthrough = []

        for i in range(x_poly.shape[1]):
            if self._is_numeric_binary(x_poly[:, i]):
                passthrough.append(i)
            else:
                continuous.append(i)

        self.continuous_indices = continuous
        self.passthrough_indices = passthrough

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "spline",
                    Pipeline(
                        [
                            ("spline", SplineTransformer(n_knots=3, degree=self.spline_degree, include_bias=False)),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    continuous,
                ),
                ("passthrough", Pipeline([("scaler", StandardScaler())]), passthrough),
            ]
        )

        self.preprocessor.fit(x_poly)
        return self

    def transform(self, x):
        x = np.asarray(x)
        x_poly = self.poly.transform(x)
        x_poly = x_poly[:, ~self.purge_cols]
        x_splined = self.preprocessor.transform(x_poly)
        x_out = np.concatenate([x_splined, np.ones((x_poly.shape[0], 1))], axis=1)

        return x_out


def ATEfunctional(data, evaluator):
    data_treated = np.copy(data)
    data_treated[:, 0] = 1
    ate_treated = evaluator(data_treated)

    data_untreated = np.copy(data)
    data_untreated[:, 0] = 0
    ate_untreated = evaluator(data_untreated)

    return ate_treated - ate_untreated


class LassoRiesz:
    def __init__(self, functional, spline_degree=3, monomial_degree=2, rL =0.01):
        self.expander = CovariateExpander(spline_degree, monomial_degree)
        self.functional = functional
        self.rho = None
        self.rL = rL

    def fit(self, data):
        x = np.concatenate([data.treatments, data.covariates],axis = 1)
        self.expander.fit(x)
        xb = self.expander.transform(x)
        mb = self.functional(x, self.expander.transform)
        hatM = np.mean(mb, axis=0)
        hatG = 1 / xb.shape[0] * (xb.T @ xb)
        rho = md_lasso(hatG, hatM, self.rL)
        self.rho = rho

    def predict(self, x):
        xb = self.expander.transform(x)
        return xb @ self.rho

    def get_riesz_representer(self,data):
        return self.predict(np.concatenate([data.treatments, data.covariates], axis = 1))