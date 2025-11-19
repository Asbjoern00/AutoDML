import numpy as np
from average_treatment_effect.dataset import Dataset


class TestDataset:
    def test_can_calculate_empirical_average_treatment_effect(self):
        dataset = Dataset(
            treatments=np.zeros((5, 1)),
            outcomes=np.zeros((5, 1)),
            counterfactual_outcomes=np.zeros((5, 1)),
            noiseless_untreated_outcomes=np.array([1, 0, 1, 4, 5]),
            noiseless_treated_outcomes=np.array([2, 1, 2, 5, 6]),
            covariates=np.zeros((5, 5)),
        )
        assert dataset.get_average_treatment_effect() == 1
