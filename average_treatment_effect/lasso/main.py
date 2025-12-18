import numpy as np
from average_treatment_effect.lasso.LassoClass import LassoRiesz,ATEfunctional
from average_treatment_effect.dataset import Dataset
from average_treatment_effect.boost_model.booster_wrapper import  BoosterWrapper

K = 1000
M = 1  # n crossfits
truths = np.zeros(K)
estimates = np.zeros(K)
estimates_OS = np.zeros(K)
ci_lower = np.zeros(K)
ci_upper = np.zeros(K)
cvg = np.zeros(K)

for i in range(K):
    data = Dataset.load_chernozhukov_replication(i + 1)

    base_booster = BoosterWrapper(LassoRiesz(ATEfunctional, spline_degree=3,monomial_degree=1, rL=0.001))
    base_booster.fit(data)



    estimates[i] = base_booster.get_ate(data)
    truths[i] = data.get_average_treatment_effect()
    estimates_OS[i] = base_booster.get_plugin_ate(data)
    print(f"i = {i}, OS = {np.mean(np.abs(truths[:i+1]-estimates_OS[:i+1]))}, Plugin = {np.mean(np.abs(truths[:i+1]-estimates[:i+1]))}")
