import numpy as np
import torch
import pytest


from average_treatment_effect.dataset import Dataset
from dataclasses import fields


@pytest.fixture
def dataset():
    return Dataset(
        treatments=np.array([0, 1, 2, 3, 4]).reshape(-1, 1),
        outcomes=np.array([0, 1, 2, 3, 4]).reshape(-1, 1),
        counterfactual_outcomes=np.array([0, 1, 2, 3, 4]).reshape(-1, 1),
        noiseless_untreated_outcomes=np.array([0, 1, 2, 3, 4]).reshape(-1, 1),
        noiseless_treated_outcomes=np.array([1, 2, 3, 4, 5]).reshape(-1, 1),
        covariates=np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]),
    )


class TestDataset:
    def test_has_correct_attributes(self, dataset):
        _fields = fields(dataset)
        expected_fields = [
            "treatments",
            "outcomes",
            "counterfactual_outcomes",
            "noiseless_untreated_outcomes",
            "noiseless_treated_outcomes",
            "covariates",
            "folds",
        ]
        assert len(_fields) == len(expected_fields)
        assert [field.name for field in _fields] == expected_fields

    def test_can_calculate_empirical_average_treatment_effect(self, dataset):
        assert dataset.get_average_treatment_effect() == 1

    def test_can_load_csv(self, dataset):
        path = "average_treatment_effect/tests/data/test_data.csv"
        dataset_from_csv = Dataset.from_csv(path)

        np.testing.assert_array_equal(dataset_from_csv.treatments, dataset.treatments)
        np.testing.assert_array_equal(dataset_from_csv.outcomes, dataset.outcomes)
        np.testing.assert_array_equal(dataset_from_csv.counterfactual_outcomes, dataset.counterfactual_outcomes)
        np.testing.assert_array_equal(
            dataset_from_csv.noiseless_untreated_outcomes, dataset.noiseless_untreated_outcomes
        )
        np.testing.assert_array_equal(dataset_from_csv.noiseless_treated_outcomes, dataset.noiseless_treated_outcomes)
        np.testing.assert_array_equal(dataset_from_csv.covariates, dataset.covariates)

    def test_can_convert_to_tensor(self, dataset):
        outcomes_tensor = dataset.get_as_tensor("outcomes")
        assert isinstance(outcomes_tensor, torch.Tensor)
        assert outcomes_tensor.shape == (5, 1)

    def test_splits_into_folds(self, dataset):
        dataset.split_into_folds(3)
        folds = dataset.folds
        indices_in_folds = np.concatenate(folds)
        indices_in_folds.sort()
        assert len(folds) == 3
        np.testing.assert_array_equal(indices_in_folds, np.array([0, 1, 2, 3, 4]))

    def test_selects_correct_folds(self, dataset):
        dataset.folds = [np.array([0, 4]), np.array([1, 2]), np.array([3])]
        first_two_folds = dataset.get_folds([1, 2])
        third_fold = dataset.get_folds([3])
        np.testing.assert_array_equal(first_two_folds.treatments, np.array([0, 4, 1, 2]).reshape(-1, 1))
        np.testing.assert_array_equal(third_fold.outcomes, np.array([3]).reshape(-1, 1))
