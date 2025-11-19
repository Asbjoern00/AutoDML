import numpy as np
from average_treatment_effect.dataset import Dataset
from dataclasses import fields


class TestDataset:
    def test_has_correct_attributes(self):
        dataset = Dataset()
        _fields = fields(dataset)
        expected_fields = [
            "treatments",
            "outcomes",
            "counterfactual_outcomes",
            "noiseless_untreated_outcomes",
            "noiseless_treated_outcomes",
            "covariates",
        ]
        assert len(_fields) == len(expected_fields)
        assert [field.name for field in _fields] == expected_fields

    def test_can_calculate_empirical_average_treatment_effect(self):
        dataset = Dataset(
            noiseless_untreated_outcomes=np.array([1, 0, 1, 4, 5]),
            noiseless_treated_outcomes=np.array([2, 1, 2, 5, 6]),
        )
        assert dataset.get_average_treatment_effect() == 1
