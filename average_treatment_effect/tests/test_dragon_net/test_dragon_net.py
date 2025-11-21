from average_treatment_effect.dragon_net.dragon_net import DragonNet


class TestDragonNet:
    def test_dragon_net_has_correct_number_of_parameters(self):
        model = DragonNet()
        number_of_parameters = sum(p.numel() for p in model.parameters())
        expected_number_of_parameters = 2 * 30301 + 85600 + 200 + 1 + 1
        assert number_of_parameters == expected_number_of_parameters
