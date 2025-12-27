from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.neighbors import KernelDensity


class ResidualKDE:
    def __init__(self, model):
        self.model = model
        self.kde = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": np.linspace(0.05, 1.0, 20)}, cv=5)
        self.shift = 1.0

    def fit(self, data):
        epsilon_hat = np.zeros(data.treatments.shape[0])
        n_folds = 2
        folds = data.split_into_folds(n_folds)
        n_evaluated = 0

        for j in range(n_folds):
            eval_data, train_data = data.get_fit_and_train_folds(folds, j)
            n_eval_data = eval_data.treatments.shape[0]
            covariates = train_data.covariates
            treatments = train_data.treatments
            self.model.fit(covariates, treatments)
            epsilon_hat[n_evaluated : n_evaluated + n_eval_data] = eval_data.treatments - self.model.predict(eval_data.covariates)
            n_evaluated += n_eval_data

        self.model.fit(data.covariates, data.treatments) #refit on entire sample
        self.kde.fit(epsilon_hat.reshape(-1, 1))

    def get_riesz_representer(self, data):
        covariates = data.covariates
        treatments = data.treatments

        fitted = self.model.predict(covariates)

        non_shifted_density = np.exp(self.kde.score_samples((treatments - fitted).reshape(-1, 1)))
        shifted_density = np.exp(self.kde.score_samples((treatments - self.shift - fitted).reshape(-1, 1)))
        ratio = shifted_density / non_shifted_density
        return ratio - 1
