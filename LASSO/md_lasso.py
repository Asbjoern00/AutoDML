import numpy as np
from numba import njit


@njit
def md_lasso(G, M, rL, D=None, rho_init=None, max_iter=200, tol=1e-3):
    p = M.shape[0]

    if D is None:
        D = np.ones(p)

    if rho_init is None:
        rho = np.zeros(p)
    else:
        rho = rho_init.copy()

    z = np.diag(G)
    Grho = G @ rho  # cache G rho

    for it in range(max_iter):
        max_change = 0.0

        for j in range(p):
            # π_j = M_j − (Gρ)_j + G_jj ρ_j
            pi_j = M[j] - Grho[j] + z[j] * rho[j]

            thresh = D[j] * rL

            if pi_j < -thresh:
                new_rj = (pi_j + thresh) / z[j]
            elif pi_j > thresh:
                new_rj = (pi_j - thresh) / z[j]
            else:
                new_rj = 0.0

            delta = new_rj - rho[j]
            if delta != 0.0:
                rho[j] = new_rj
                Grho += G[:, j] * delta
                max_change = max(max_change, abs(delta))
        # print(it, max_change)

        if max_change < tol:
            break

    return rho


def compute_loadings(xb, mb, rho, c3=0.1, intercept_indices = np.array([0])):
    # shape: (n,)
    t = xb @ rho
    # shape: (n,p)
    residual = xb * t[:, None] - mb
    hatD = np.mean(residual * residual, axis=0) + 0.2 # 0.2 for stability according to Chernuzhukov
    hatD[intercept_indices] *= c3
    return np.sqrt(hatD)