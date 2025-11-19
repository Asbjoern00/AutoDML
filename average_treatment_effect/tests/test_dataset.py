import numpy as np
import torch

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
            'folds'
        ]
        assert len(_fields) == len(expected_fields)
        assert [field.name for field in _fields] == expected_fields

    def test_can_calculate_empirical_average_treatment_effect(self):
        dataset = Dataset(
            noiseless_untreated_outcomes=np.array([1, 0, 1, 4, 5]),
            noiseless_treated_outcomes=np.array([2, 1, 2, 5, 6]),
        )
        assert dataset.get_average_treatment_effect() == 1

    def test_can_load_csv(self):
        path = "average_treatment_effect/tests/data/test_data.csv"
        treatments = np.array([1, 0, 1]).reshape(-1, 1)
        outcomes = np.array([0, 1, 2]).reshape(-1, 1)
        counterfactual_outcomes = np.array([3, 4, 5]).reshape(-1, 1)
        noiseless_untreated_outcomes = np.array([6, 7, 8]).reshape(-1, 1)
        noiseless_treated_outcomes = np.array([9, 1, 2]).reshape(-1, 1)
        covariates = np.array([[0, 3], [1, 4], [2, 5]])

        dataset = Dataset.from_csv(path)
        assert (dataset.treatments == treatments).all()
        assert (dataset.outcomes == outcomes).all()
        assert (dataset.counterfactual_outcomes == counterfactual_outcomes).all()
        assert (
            dataset.noiseless_untreated_outcomes == noiseless_untreated_outcomes
        ).all()
        assert (dataset.noiseless_treated_outcomes == noiseless_treated_outcomes).all()
        assert (dataset.covariates == covariates).all()

    def test_can_convert_to_tensor(self):
        dataset = Dataset(outcomes=np.array([0, 1, 2]).reshape(-1, 1))
        outcomes_tensor = dataset.get_as_tensor("outcomes")
        assert isinstance(outcomes_tensor, torch.Tensor)
        assert outcomes_tensor.shape == (3, 1)
