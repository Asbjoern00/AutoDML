import torch
import numpy as np
from dope_neural_nets.model import ModelWrapper
from dope_neural_nets.dataset import Dataset

np.random.seed(1)


def run_experiment():
   data = Dataset.simulate_dataset(1000, 10)
   model_wrapper = ModelWrapper()
   folds = data.split_into_folds(5)
   estimate_components = []
   for j in range(5):
       fit_fold, train_folds = Dataset.get_fit_and_train_folds(folds, j)
       model_wrapper.train_outcome_head(train_folds, train_shared_layers=True)
       model_wrapper.train_riesz_head(train_folds, train_shared_layers=False)
       estimate_components.append(model_wrapper.get_estimate_components(fit_fold))
   estimate_components = torch.concat(estimate_components, dim=0)
   estimate = torch.mean(estimate_components).item()
   variance = torch.var(estimate_components).item()
   return {'estimate': estimate, 'variance': variance}
print(run_experiment())