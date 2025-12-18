import torch
from ihdp_average_treatment_effect import dataset
from ihdp_average_treatment_effect.dragon_net.dragon_net_wrapper import DragonNetWrapper

torch.manual_seed(42)
estimates = []
truths = []
for i in range(1000):
    print("Iteration", i + 1)
    data = dataset.Dataset.load_chernozhukov_replication(i + 1)
    truth = data.get_average_treatment_effect()
    truths.append(truth)
    dragon_net = DragonNetWrapper.create()
    dragon_net.train(data)
    estimate = dragon_net.get_average_treatment_effect(data)
    estimates.append(estimate)
    MAE = sum(abs(est - truth) for est, truth in zip(estimates, truths)) / len(estimates)
    print("estimate", estimate)
    print("Truth:", truth)
    print("MAE:", MAE)
