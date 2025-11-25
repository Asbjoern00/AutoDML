import numpy as np

from average_treatment_effect.boost_model.riesz_model import RieszBooster
from average_treatment_effect.dataset import Dataset
from average_treatment_effect.boost_model.outcome_model import OutcomeBooster
from average_treatment_effect.boost_model.treatment_model import TreatmentBooster

ests0 = []
ests1 = []
plugs = []
truths = []

folds = 10

for i in range(1000):
    data = Dataset.load_chernozhukov_replication(i + 1)
    data.split_into_folds(folds)
    truth = data.get_average_treatment_effect()
    truths.append(truth)
    plugin_est = 0
    dr_est0 = 0
    dr_est1 = 0

    for j in range(folds):
        in_fold = data.get_folds([j + 1])
        out_of_fold = data.get_folds([k + 1 for k in range(folds) if k != j])
        outcome_model = OutcomeBooster()
        outcome_model.fit(out_of_fold, boost_round0=275, boost_round1=40)
        plugin = outcome_model.get_plugin_estimate(in_fold)
        plugin_est += plugin / folds
        residuals = outcome_model.get_residuals(in_fold)

        treatment_model = TreatmentBooster()
        treatment_model.fit(out_of_fold, num_boost_round=100)
        propensity_riesz_representer = treatment_model.get_riesz_representer(in_fold)

        riesz_boost_model = RieszBooster()
        riesz_boost_model.fit(out_of_fold, num_boost_round=80)
        direct_riesz_representer = riesz_boost_model.get_riesz_representer(in_fold)

        correction0 = np.mean(residuals * propensity_riesz_representer)
        dr_est0 += (plugin + correction0) / folds

        correction1 = np.mean(residuals * direct_riesz_representer)
        dr_est1 += (plugin + correction1) / folds

    plugs.append(plugin_est)
    MAE_plug = sum(np.abs(np.array(truths) - np.array(plugs))) / len(truths)
    ests0.append(dr_est0)
    MAE0 = sum(np.abs(np.array(truths) - np.array(ests0))) / len(truths)
    ests1.append(dr_est1)
    MAE1 = sum(np.abs(np.array(truths) - np.array(ests1))) / len(truths)

    print(plugin_est, dr_est0, dr_est1, truth)
    print("Iteration:", i + 1, "MAE0:", MAE0, "MAE1:", MAE1, "MAEPlug", MAE_plug)
