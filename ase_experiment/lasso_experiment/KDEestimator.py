from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neighbors import KernelDensity


class ResidualKDE:
    def __init__(self, model):
        self.model = model
        self.kde = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": np.linspace(0.05, 1.0, 20)}, cv=5)
        self.shift = 1.0

    def fit(self, data):
        covariates = data.covariates
        treatments = data.treatments

        self.model.fit(
            covariates, treatments
        )  # consider splitting data here so as to not compute the residual on the same data as it is fitted on
        epsilon_hat = treatments - self.model.predict(covariates)
        self.kde.fit(epsilon_hat.reshape(-1, 1))

    def get_riesz_representer(self, data):
        covariates = data.covariates
        treatments = data.treatments

        fitted = self.model.predict(covariates)

        non_shifted_density = np.exp(self.kde.score_samples((treatments - fitted).reshape(-1, 1)))
        shifted_density = np.exp(self.kde.score_samples((treatments - self.shift - fitted).reshape(-1, 1)))
        return shifted_density / non_shifted_density - 1
