import numpy as np

from average_treatment_effect import RieszNetBase
from average_treatment_effect import dataset

K = 1000
M = 5  # n crossfits
truths = np.zeros(K)
estimates = np.zeros(K)
estimates_OS = np.zeros(K)

for i in range(K):
    data = dataset.Dataset.load_chernozhukov_replication(i + 1)
    data.split_into_folds(M)
    splits = np.arange(1, M + 1)
    rr_module = RieszNetBase.RieszNetBaseModule(RieszNetBase.ate_functional)
    for j in range(M):
        rr_module.train(data = data.get_folds(np.delete(splits, j)))
        estimates_rnet = rr_module.get_estimate(data = data.get_folds([splits[j]]))
        estimates_OS[i] += 1 / M * estimates_rnet["one step estimate"]
        estimates[i] += 1 / M * estimates_rnet["plugin"]

    truths[i] = data.get_average_treatment_effect()

    print(i)
    print("running MAE OS: {}".format(np.mean(np.abs(truths[: i + 1] - estimates_OS[: i + 1]))))
    print("running MAE plugin: {}".format(np.mean(np.abs(truths[: i + 1] - estimates[: i + 1]))))


np.save("truths.npy", truths)
np.save("estimates.npy", estimates)
np.save("estimates_OS.npy", estimates_OS)

print(f"MAE = {np.mean(np.abs(truths - estimates))}")
print(f"MAE_OS = {np.mean(np.abs(truths - estimates_OS))}")
