import torch

from average_treatment_effect import dragon_net


class TestDragonNet:
    def test_shared_net_has_correct_number_of_parameters(self):
        model = dragon_net.SharedNet()
        number_of_parameters = sum(p.numel() for p in model.parameters())
        expected_number_of_parameters = 85600
        assert number_of_parameters == expected_number_of_parameters

    def test_outcome_net_has_correct_number_of_parameters(self):
        model = dragon_net.OutcomeNet()
        number_of_parameters = sum(p.numel() for p in model.parameters())
        expected_number_of_parameters = 30301
        assert number_of_parameters == expected_number_of_parameters

    def test_base_dragon_net_has_correct_number_of_parameters(self):
        model = dragon_net.BaseDragonNet()
        number_of_parameters = sum(p.numel() for p in model.parameters())
        expected_number_of_parameters = 2 * 30301 + 85600 + 200 + 1
        assert number_of_parameters == expected_number_of_parameters


class TestDragonNetLoss:
    def test_base_dragon_net_loss_returns_expected(self):
        criterion = dragon_net.BaseDragonNetLoss(cross_entropy_weight=0.5)
        treatments = torch.tensor([0, 1, 1]).reshape(-1, 1).float()
        treatment_predictions = torch.tensor([0.3, 0.7, 0.6]).reshape(-1, 1)
        outcomes = torch.tensor([11, 1, -1]).reshape(-1, 1)
        outcome_predictions = torch.tensor([8, 0.7, 0.6]).reshape(-1, 1)
        expected_mse_loss = torch.mean((outcomes - outcome_predictions) ** 2)
        expected_cross_entropy_loss = torch.mean(
            -(treatments * torch.log(treatment_predictions) + (1 - treatments) * torch.log(1 - treatment_predictions))
        )
        expected_loss = expected_mse_loss * 0.5 + expected_cross_entropy_loss * 0.5
        loss = criterion(
            {"treatment_predictions": treatment_predictions, "outcome_predictions": outcome_predictions},
            treatments,
            outcomes,
        )
        assert torch.equal(loss, expected_loss)
