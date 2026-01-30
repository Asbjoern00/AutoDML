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


def redraw_Y(T, W, b):
    n = W.shape[0]
    conditional_mean_y = -0.6 * T + (W @ b).reshape(-1)
    y = conditional_mean_y + np.sqrt(5.6 * np.var(conditional_mean_y)) * np.random.normal(size=n)
    return y


def simulate_datasets_as_chernuzhukov():
    W, T = clean_dataset_as_chernuzhukov()
    mu_T, sigma2_T = estimate_cond_mean_variance(W, T)
    n,p = W.shape
    k = 0
    headers = ["Truth", "Y", "T"] + [f"W_{i+1}" for i in range(p)]
    for i in range(10):
        np.random.seed(i)
        b = np.random.uniform(-0.5, 0.5, size=(p, 1))
        for j in range(100):
            T_redrawn = redraw_treatment(W, mu_T, sigma2_T)
            Y_redrawn = redraw_Y(T_redrawn, W, b)
            out = np.concatenate([np.repeat(-0.6, n).reshape(-1,1), Y_redrawn.reshape(-1, 1), T_redrawn.reshape(-1, 1), W], axis=1)
            np.savetxt(
                f"AveragePartialDerivative/BHP_data/redrawn_datasets/data_{k}.csv",
                out,
                delimiter=",",
                header=",".join(headers),
                comments="",
            )
            k = k + 1


if __name__ == "__main__":
    simulate_datasets_as_chernuzhukov()
