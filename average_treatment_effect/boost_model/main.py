import numpy as np

from average_treatment_effect.boost_model.booster_wrapper import BoosterWrapper
from average_treatment_effect.dataset import Dataset

np.random.seed(42)

base_estimates = []
riesz_estimates = []
plugs = []
truths = []

folds = 10

for i in range(1000):
    data = Dataset.load_chernozhukov_replication(i + 1)
    data.split_into_folds(folds)
    truth = data.get_average_treatment_effect()
    truths.append(truth)
    dr_est_base = 0
    dr_est_riesz = 0
    n_total = data.treatments.shape[0]

    for j in range(folds):
        in_fold = data.get_folds([j + 1])
        out_of_fold = data.get_folds([k + 1 for k in range(folds) if k != j])
        n_in_fold = in_fold.treatments.shape[0]

        base_booster = BoosterWrapper.create_base_booster()
        base_booster.fit(out_of_fold)
        base_ate = base_booster.get_ate(in_fold)
        dr_est_base += base_ate * n_in_fold / n_total

        riesz_booster = BoosterWrapper.create_riesz_booster()
        riesz_booster.fit(out_of_fold)
        riesz_ate = riesz_booster.get_ate(in_fold)
        dr_est_riesz += riesz_ate * n_in_fold / n_total

    base_estimates.append(dr_est_base)
    mae_base = sum(np.abs(np.array(truths) - np.array(base_estimates))) / len(truths)
    riesz_estimates.append(dr_est_riesz)
    riesz_base = sum(np.abs(np.array(truths) - np.array(riesz_estimates))) / len(truths)

    print(dr_est_base, dr_est_riesz, truth)
    print("Iteration:", i + 1, "MAE0:", mae_base, "MAE1:", riesz_base)
