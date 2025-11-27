import numpy as np

from average_treatment_effect.boost_model.outcome_model import OutcomeBooster
from average_treatment_effect.boost_model.treatment_model import TreatmentBooster
from average_treatment_effect.boost_model.riesz_model import RieszBooster


class BoosterWrapper:
    def __init__(self, riesz_rep_model):
        self.riesz_rep_model = riesz_rep_model
        self.outcome_model = OutcomeBooster()

    @classmethod
    def create_base_booster(cls):
        return cls(TreatmentBooster())

    @classmethod
    def create_riesz_booster(cls):
        return cls(RieszBooster())

    def fit(self, data):
        self.outcome_model.fit(data)
        self.riesz_rep_model.fit(data)

    def get_ate(self, data):
        return np.mean(self.get_individual_treatment_effects(data))

    def get_individual_treatment_effects(self, data):
        functional = self.outcome_model.get_functional(data)
        residuals = self.outcome_model.get_residuals(data)
        riesz_rep = self.riesz_rep_model.get_riesz_representer(data)
        correction = residuals * riesz_rep
        return functional + correction

    def get_variance(self, data):
        individual_effects = self.get_individual_treatment_effects(data)
        ate = self.get_ate(data)
        return np.mean((individual_effects - ate) ** 2)

    def get_plugin_ate(self, data):
        functional = self.outcome_model.get_functional(data)
        return np.mean(functional)

    def get_plugin_variance(self, data):
        functional = self.outcome_model.get_functional(data)
        plugin = self.get_plugin_ate(data)
        return np.mean((functional - plugin) ** 2)
