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
