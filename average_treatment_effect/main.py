from average_treatment_effect import dragon_net
from average_treatment_effect import dataset

data = dataset.Dataset.load_chernozhukov_replication(1)
dragon_net_module = dragon_net.DragonNetModule()
dragon_net_module.train(epochs=500, data=data)


estimates = dragon_net_module.get_average_treatment_effect_estimate(data)
print(data.get_average_treatment_effect())
print(estimates)
