import numpy as np

from average_treatment_effect.boost_model.riesz_model import RieszBooster
from average_treatment_effect.dataset import Dataset
from average_treatment_effect.boost_model.outcome_model import OutcomeBooster
from average_treatment_effect.boost_model.treatment_model import TreatmentBooster

ests0 = []
ests1 = []
truths = []

data = Dataset.load_chernozhukov_replication(1)


treatment_model = TreatmentBooster()
treatment_model.fit(data, num_boost_round=1000)
propensity_riesz_representer = treatment_model.get_riesz_representer(data)

riesz_representer_model = RieszBooster()
riesz_representer_model.fit(data, num_boost_round=1000)
direct_riesz_representer = riesz_representer_model.get_riesz_representer(data)

for i in range(1000):
    data = Dataset.load_chernozhukov_replication(i + 1)
    truth = data.get_average_treatment_effect()
    truths.append(truth)

    outcome_model = OutcomeBooster()
    outcome_model.fit(data, num_boost_round=300)
    plugin = outcome_model.get_plugin_estimate(data)
    residuals = outcome_model.get_residuals(data)

    correction0 = np.mean(residuals * propensity_riesz_representer)
    dr_est0 = plugin + correction0
    ests0.append(dr_est0)
    MAE0 = sum(np.abs(np.array(truths) - np.array(ests0))) / len(truths)

    correction1 = np.mean(residuals * direct_riesz_representer)
    dr_est1 = plugin + correction1
    ests1.append(dr_est1)
    MAE1 = sum(np.abs(np.array(truths) - np.array(ests1))) / len(truths)

    print(plugin, dr_est0, dr_est1, truth)
    print("Iteration:", i + 1, "MAE0:", MAE0, "MAE1:", MAE1)
