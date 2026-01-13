import torch
import numpy as np
from dope_neural_nets.model import ModelWrapper
from dope_neural_nets.dataset import Dataset

np.random.seed(1)
torch.manual_seed(1)

truth = 2.121539888279284


def run_experiment():
    data = Dataset.simulate_dataset(1000, 10)
    folds = data.split_into_folds(5)
    estimate_components = []
    for j in range(5):
        fit_fold, train_folds = Dataset.get_fit_and_train_folds(folds, j)
        model_wrapper = ModelWrapper()
        model_wrapper.train_as_riesz_net(train_folds, rr_w=0.1)
        estimate_components.append(model_wrapper.get_estimate_components(fit_fold))
    estimate_components = torch.concat(estimate_components, dim=0)
    estimate = torch.mean(estimate_components).item()
    variance = torch.var(estimate_components).item()
    return {"estimate": estimate, "variance": variance}


estimates = []
for i in range(1000):
    result = run_experiment()
    estimates.append(result["estimate"])
    residuals = [np.abs(est-truth) for est in estimates if np.abs(est-truth) <= 0.5]
    mse = sum(residual**2 for residual in residuals) / len(residuals)
    print(i, "Estimate:", result['estimate'], "RMSE:", mse**0.5)