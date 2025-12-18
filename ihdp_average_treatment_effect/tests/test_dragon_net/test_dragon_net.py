import torch
from ihdp_average_treatment_effect.dragon_net.dragon_net import DragonNet, DragonNetLoss


class TestDragonNet:
    def test_dragon_net_has_correct_number_of_parameters(self):
        model = DragonNet()
        number_of_parameters = sum(p.numel() for p in model.parameters())
        expected_number_of_parameters = 2 * 30301 + 85600 + 200 + 1 + 1
        assert number_of_parameters == expected_number_of_parameters


class TestDragonNetLoss:
    def test_dragon_net_loss_returns_expected(self):
        criterion = DragonNetLoss(outcome_mse_weight=0.5, treatment_cross_entropy_weight=0.2, tmle_weight=0.3)
        treatments = torch.tensor([0, 1, 1]).reshape(-1, 1).float()
        treatment_predictions = torch.tensor([0.3, 0.7, 0.6]).reshape(-1, 1)
        outcomes = torch.tensor([11, 1.1, -1]).reshape(-1, 1)
        base_outcome_predictions = torch.tensor([8, 0.7, 0.6]).reshape(-1, 1)
        targeted_outcome_predictions = torch.tensor([7, 0.5, 0.6]).reshape(-1, 1)
        expected_mse_loss = torch.mean((outcomes - base_outcome_predictions) ** 2)
        expected_cross_entropy_loss = torch.mean(
            -(treatments * torch.log(treatment_predictions) + (1 - treatments) * torch.log(1 - treatment_predictions))
        )
        expected_tmle_loss = torch.mean((outcomes - targeted_outcome_predictions) ** 2)
        expected_loss = expected_mse_loss * 0.5 + expected_cross_entropy_loss * 0.2 + expected_tmle_loss * 0.3
        loss = criterion(
            {
                "treatment_prediction": treatment_predictions,
                "base_outcome_prediction": base_outcome_predictions,
                "targeted_outcome_prediction": targeted_outcome_predictions,
            },
            treatments=treatments,
            outcomes=outcomes,
        )
        assert torch.equal(loss, expected_loss)
