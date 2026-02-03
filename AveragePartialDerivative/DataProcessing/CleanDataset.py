import numpy as np
import pandas as pd
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict


def clean_dataset_as_chernuzhukov():
    df = pd.read_csv("AveragePartialDerivative/BHP_data/raw_data.csv")
    df = df[df["log_p"] > math.log(1.2)]
    df = df[df["log_y"] > math.log(15000)]
    Xdf = df.iloc[:, 1:]
    state_dum = pd.get_dummies(Xdf["state_fips"], prefix="state")
    Xdf = pd.concat([Xdf, state_dum], axis=1)
    Xdf = Xdf.drop(["distance_oil1000", "state_fips", "share"], axis=1)
    W = Xdf.drop(["log_p"], axis=1).values
    T = Xdf["log_p"].values

    return W, T


def estimate_cond_mean_variance(W, T):
    mu_T = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, random_state=123)
    mu_T.fit(W, T)

    # Conditional Variance
    sigma2_T = RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_depth=5, random_state=123)
    e_T = T - cross_val_predict(mu_T, W, T)
    sigma2_T.fit(W, e_T**2)
    return mu_T, sigma2_T


def redraw_treatment(W, mu_T, sigma2_T):
    n_samples = W.shape[0]
    predicted_mean = mu_T.predict(W)
    predicted_var = sigma2_T.predict(W)
    T_redrawn = np.random.normal(loc=predicted_mean, scale=np.sqrt(predicted_var), size=n_samples)
    return T_redrawn


def redraw_Y(cond_mean_fn, T, W, b, c):
    n = W.shape[0]
    conditional_mean_y = cond_mean_fn(T,W,b,c)
    y = conditional_mean_y + np.sqrt(5.6 * np.var(conditional_mean_y)) * np.random.normal(size=n)
    return y


def nl(W):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    W = W.astype(np.float32)
    return 1.5 * sigmoid(10 * W[:, 5]) + 1.5 * sigmoid(10 * W[:, 7])


def simple_f_with_linear_confounders(T, W, b, c):
    conditional_mean_y = -0.6 * T + (W @ b).reshape(-1)
    return conditional_mean_y


def complex_f_with_linear_and_non_linear_confounders(T, W, b, c):
    conditional_mean_y = (
        -1 / 6 * T ** (3) * (W[:, 0] ** 2 * 1 / 10 + (W[:, 0:8] @ c).reshape(-1))
        + (W @ b).reshape(-1)
        + nl(W)
    )

    return conditional_mean_y

def derivative_complex_f_with_linear_and_non_linear_confounders(T,W,b,c):
    derivative_cond_mean = -1 / 2 * T ** (2) * (W[:, 0] ** 2 * 1 / 10 + (W[:, 0:8] @ c).reshape(-1))
    return derivative_cond_mean

def derivative_simple_f_with_linear_confounders(T,W,b,c):
    return np.repeat(-0.6, T.shape[0])

def simulate_datasets_as_chernuzhukov():
    W, T = clean_dataset_as_chernuzhukov()
    mu_T, sigma2_T = estimate_cond_mean_variance(W, T)
    n, p = W.shape
    k = 0
    headers = ["Truth", "Y", "T"] + [f"W_{i+1}" for i in range(p)]

    cond_mean_fns = [simple_f_with_linear_confounders, complex_f_with_linear_and_non_linear_confounders]
    cond_mean_derivs = [derivative_simple_f_with_linear_confounders, derivative_complex_f_with_linear_and_non_linear_confounders]


    for i in range(10):
        np.random.seed(i)
        b = np.random.uniform(-0.5, 0.5, size=(p, 1))
        c = np.random.uniform(-0.2, 0.2, size=(8, 1))
        for j in range(100):
            T_redrawn = redraw_treatment(W, mu_T, sigma2_T)

            for cond_mean_fn,deriv_cond_mean in zip(cond_mean_fns, cond_mean_derivs):
                Y_redrawn = redraw_Y(cond_mean_fn, T_redrawn, W, b,c)
                derivs = deriv_cond_mean(T_redrawn,W,b,c)
                out = np.concatenate(
                    [derivs.reshape(-1, 1), Y_redrawn.reshape(-1, 1), T_redrawn.reshape(-1, 1), W], axis=1
                )
                np.savetxt(
                    f"AveragePartialDerivative/BHP_data/redrawn_datasets/{cond_mean_fn.__name__}/data_{k}.csv",
                    out,
                    delimiter=",",
                    header=",".join(headers),
                    comments="",
                )
                k = k + 1
if __name__ == "__main__":
    simulate_datasets_as_chernuzhukov()
