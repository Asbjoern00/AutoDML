import numpy as np

from average_treatment_effect.dataset import Dataset
from average_treatment_effect.boost_model.outcome_model import OutcomeBooster
from average_treatment_effect.boost_model.treatment_model import TreatmentBooster

ests = []
truths = []

for i in range(1000):
    data = Dataset.load_chernozhukov_replication(i + 1)
    truth = data.get_average_treatment_effect()
    truths.append(truth)

    outcome_model = OutcomeBooster()
    outcome_model.fit(data, num_boost_round=500)
    plugin = outcome_model.get_plugin_estimate(data)
    residuals = outcome_model.get_residuals(data)

    treatment_model = TreatmentBooster()
    treatment_model.fit(data, num_boost_round=500)
    riesz_representer = treatment_model.get_riesz_representer(data)

    correction = np.mean(residuals * riesz_representer)
    dr_est = plugin + correction
    print(plugin, dr_est, truth)
    ests.append(plugin)
    MAE = sum(np.abs(np.array(truths) - np.array(ests))) / len(truths)
    print("MAE:", MAE)
