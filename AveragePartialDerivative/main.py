import numpy as np

from AveragePartialDerivative import dataset
from average_treatment_effect.riesz_net import RieszNetBaseModule
from AveragePartialDerivative.AveragePartialDerivativeFunctional import avg_der_functional

K = 1000
M = 5  # n crossfits
truths = np.zeros(K)
# estimates = np.zeros(K)
estimates_OS = np.zeros(K)
ci_lower = np.zeros(K)
ci_upper = np.zeros(K)
cvg = np.zeros(K)

for i in range(K):
    data = dataset.Dataset.from_sample(n=1000)
    rr_module = RieszNetBaseModule.RieszNetBaseModule(avg_der_functional, epochs=300,in_features=data.covariates.shape[1]+1,tmle_weight=0.1)
    fitted = rr_module.fit(data,n_crossfit=M)


    estimates_OS[i] = fitted["one step estimate"]
    print(estimates_OS[i])
    ci_lower[i] = estimates_OS[i]-1.96*fitted["std_error"]
    ci_upper[i] = estimates_OS[i] + 1.96 * fitted["std_error"]
    truths[i] = data.avg_partial_derivative
    cvg[i] = (ci_lower[i] < truths[i]) * (truths[i] < ci_upper[i])

    print(i)
    print("running MAE OS: {}".format(np.mean(np.abs(truths[: i + 1] - estimates_OS[: i + 1]))))
    print(f"running CVG: {np.mean(cvg[: i + 1])}")
    #print("running MAE plugin: {}".format(np.mean(np.abs(truths[: i + 1] - estimates[: i + 1]))))


#np.save("truths.npy", truths)
#np.save("estimates_OS.npy", estimates_OS)

#print(f"MAE = {np.mean(np.abs(truths - estimates))}")
#print(f"MAE_OS = {np.mean(np.abs(truths - estimates_OS))}")
