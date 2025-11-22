from average_treatment_effect import base_dragon_net
from average_treatment_effect import RieszNetBase
from average_treatment_effect import dataset

data = dataset.Dataset.load_chernozhukov_replication(1)
rr_module = RieszNetBase.RieszNetBaseModule(RieszNetBase.ate_functional)
rr_module.tune_weight_decay(data)
rr_module.train(data)
estimates_rnet = rr_module.get_estimate(data)


dragon_net_module = base_dragon_net.BaseDragonNetModule()
dragon_net_module.train(epochs=1000, data=data)


estimates_dnet = dragon_net_module.get_average_treatment_effect_estimate(data)
print(f"True :{data.get_average_treatment_effect()}")
print(f"Dragonnet : {estimates_dnet}")
print(f"Riesznet :{estimates_rnet}")
