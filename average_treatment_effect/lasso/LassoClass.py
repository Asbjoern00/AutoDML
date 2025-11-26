from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from average_treatment_effect.dataset import Dataset
from sklearn.ensemble import RandomForestRegressor
import numpy as np



def md_lasso_cd_paper(G, M, rL, D=None, rho_init=None, max_iter=5000, tol=1e-8, diag_eps=1e-12, verbose=False):
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
    G = np.asarray(G, dtype=float)
    M = np.asarray(M, dtype=float)
    p = M.size

    if D is None:
        D = np.ones(p, dtype=float)
    else:
        D = np.asarray(D, dtype=float)

    if rho_init is None:
        rho = np.zeros(p, dtype=float)
    else:
        rho = np.array(rho_init, dtype=float)

    # Precompute diagonal and check for tiny diagonals
    G_diag = np.diag(G).copy()
    small_diag = np.abs(G_diag) < diag_eps
    if np.any(small_diag):
        # don't change G itself; use eps'd values for division only
        G_diag_safe = G_diag.copy()
        G_diag_safe[small_diag] = diag_eps
    else:
        G_diag_safe = G_diag

    # Maintain current residual r = M - G @ rho  (so pi_j = M_j - sum_{k != j} rho_k G_jk = r_j + G_jj * rho_j)
    # But it's convenient to maintain "Grho = G @ rho" and compute pi_j = M[j] - (Grho[j] - G[j,j]*rho[j])
    Grho = G.dot(rho)

    for it in range(1, max_iter + 1):
        rho_old = rho.copy()

        # One full coordinate sweep (j = 0..p-1)
        for j in range(p):
            # compute pi_j = M_j - sum_{k != j} rho_k G_jk
            # sum_{k != j} rho_k G_jk = (Grho[j] - G[j,j] * rho[j])
            pi_j = M[j] - (Grho[j] - G_diag[j] * rho[j])

            if j == p-1:
                new_rj = pi_j / G_diag_safe[j] # no penalization consider if this is the right thing to do

            else:
                thresh = D[j] * rL

                if pi_j < -thresh:
                    new_rj = (pi_j + thresh) / G_diag_safe[j]
                elif pi_j > thresh:
                    new_rj = (pi_j - thresh) / G_diag_safe[j]
                else:
                    new_rj = 0.0

            # If diagonal was tiny and we adjusted, this will avoid huge jumps.
            # Update rho and incremental Grho := Grho + G[:, j] * (new_rj - rho[j])
            delta = new_rj - rho[j]
            if delta != 0.0:
                Grho += G[:, j] * delta
                rho[j] = new_rj

        # convergence check
        change = np.linalg.norm(rho - rho_old)
        if verbose and (it % 50 == 0 or change < tol):
            # compute objective value for diagnostics
            obj = 0.5 * rho.dot(G.dot(rho)) - rho.dot(M) + rL * np.sum(np.abs(D * rho))
            print(f"iter {it:4d}  change {change:.3e}  obj {obj:.6e}")
        if change < tol:
            converged = True
            break
    else:
        converged = False
        it = max_iter

    final_obj = 0.5 * rho.dot(G.dot(rho)) - rho.dot(M) + rL * np.sum(np.abs(D * rho))
    info = {"n_iter": it, "converged": converged, "final_obj": final_obj}
    return rho, info


# dat = Dataset.load_chernozhukov_replication(1)


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
                            ("spline", SplineTransformer(degree=self.spline_degree, include_bias=False)),
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
    def __init__(self, functional, spline_degree=3, monomial_degree=2):
        self.expander = CovariateExpander(spline_degree, monomial_degree)
        self.functional = functional
        self.rho = None

    def fit(self, x):
        self.expander.fit(x)
        xb = self.expander.transform(x)
        mb = self.functional(x, self.expander.transform)
        hatM = np.mean(mb, axis=0)
        hatG = 1 / xb.shape[0] * (xb.T @ xb)

        rho, _ = md_lasso_cd_paper(hatG, hatM, 0.01)
        self.rho = rho

    def predict(self, x):
        xb = self.expander.transform(x)
        return xb @ self.rho



