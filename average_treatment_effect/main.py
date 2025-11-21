from average_treatment_effect.dragon_net_wrapper import DragonNetWrapper
from average_treatment_effect import dataset

K = 1000
estimates = []
base_estimates = []
truths = []
for i in range(K):
    print(f'Iterations Remaining: {K - i}')
    data = dataset.Dataset.load_chernozhukov_replication(i + 1)
    truths.append(data.get_average_treatment_effect())

    dragon_net = DragonNetWrapper.create_dragon_net(weight_decay=1e-4)
    base_dragon_net = DragonNetWrapper.create_base_dragon_net(weight_decay=1e-4)

    dragon_net.train_model(data)
    base_dragon_net.train_model(data)

    estimates.append(dragon_net.get_average_treatment_effect(data)['plugin_estimate'])
    base_estimates.append(base_dragon_net.get_average_treatment_effect(data)['one_step_estimate'])

    dragon_net_error = sum(abs(estimate - truth) for estimate, truth in zip(estimates, truths)) / len(estimates)
    base_dragon_net_error = sum(abs(estimate - truth) for estimate, truth in zip(base_estimates, truths)) / len(base_estimates)

    print(f'DragonNet Mean Abs Error: {dragon_net_error}')
    print(f'BaseDragonNet Mean Abs Error: {base_dragon_net_error}')



