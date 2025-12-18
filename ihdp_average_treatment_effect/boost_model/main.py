import numpy as np

from ihdp_average_treatment_effect.boost_model.booster_wrapper import BoosterWrapper
from ihdp_average_treatment_effect.dataset import Dataset

np.random.seed(42)

base_estimates = []
riesz_estimates = []
base_coverages = []
riesz_coverages = []
riesz_stds = []
base_stds = []
truths = []

folds = 10

for i in range(1000):
    data = Dataset.load_chernozhukov_replication(i + 1)
    data.split_into_folds(folds)
    truth = data.get_average_treatment_effect()
    truths.append(truth)
    dr_est_base = 0
    dr_est_riesz = 0
    base_var = 0
    riesz_var = 0
    n_total = data.treatments.shape[0]

    for j in range(folds):
        in_fold = data.get_folds([j + 1])
        out_of_fold = data.get_folds([k + 1 for k in range(folds) if k != j])
        n_in_fold = in_fold.treatments.shape[0]

        base_booster = BoosterWrapper.create_base_booster()
        base_booster.fit(out_of_fold)
        base_ate = base_booster.get_ate(in_fold)
        dr_est_base += base_ate * n_in_fold / n_total
        base_var += base_booster.get_variance(data) * n_in_fold / n_total

        riesz_booster = BoosterWrapper.create_riesz_booster()
        riesz_booster.fit(out_of_fold)
        riesz_ate = riesz_booster.get_ate(in_fold)
        dr_est_riesz += riesz_ate * n_in_fold / n_total
        riesz_var += riesz_booster.get_variance(data) * n_in_fold / n_total

    base_estimates.append(dr_est_base)
    base_std = np.sqrt(base_var / n_total)
    base_stds.append(base_std)
    lower = dr_est_base - base_std * 1.96
    upper = dr_est_base + base_std * 1.96
    base_coverages.append((truth >= lower) and (truth <= upper))

    riesz_estimates.append(dr_est_riesz)
    riesz_std = np.sqrt(riesz_var / n_total)
    riesz_stds.append(riesz_std)
    lower = dr_est_riesz - riesz_std * 1.96
    upper = dr_est_riesz + riesz_std * 1.96
    riesz_coverages.append((truth >= lower) and (truth <= upper))

    mae_base = sum(np.abs(np.array(truths) - np.array(base_estimates))) / len(truths)
    base_coverage = sum(base_coverages) / len(base_coverages)
    riesz_mae = sum(np.abs(np.array(truths) - np.array(riesz_estimates))) / len(truths)
    riesz_coverage = sum(riesz_coverages) / len(riesz_coverages)


    print("Iteration:", i + 1)
    print(dr_est_base, dr_est_riesz, truth)
    print("Base MAE:", mae_base, "Riesz MAE:", riesz_mae, "Plugin MAE")
    print("Base Coverage:", base_coverage, "Riesz Coverage:", riesz_coverage)

base_errors = [(est-truth)/se for est, truth, se in zip(base_estimates, truths, base_stds)]
riesz_errors = [(est-truth)/se for est, truth, se in zip(riesz_estimates, truths, riesz_stds)]

#np.savetxt("ihdp_average_treatment_effect/boost_model/results/base_output.csv", np.array(base_errors), delimiter=",")
#np.savetxt("ihdp_average_treatment_effect/boost_model/results/riesz_output.csv", np.array(riesz_errors), delimiter=",")
