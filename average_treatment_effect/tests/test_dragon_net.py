from average_treatment_effect import dragon_net


class TestDragonNet:
    def test_shared_net_has_correct_number_of_parameters(self):
        model = dragon_net.SharedNet()
        number_of_parameters = sum(p.numel() for p in model.parameters())
        expected_number_of_parameters = 25 * 200 + 200 + 200 * 200 + 200 + 200 * 200 + 200
        assert number_of_parameters == expected_number_of_parameters
