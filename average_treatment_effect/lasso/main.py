import numpy as np
from ate_experiment.dataset_highdim import DatasetHighDim
from average_treatment_effect.lasso.OutcomeLASSO import OutcomeLASSO
from average_treatment_effect.Functional.ATEFunctional import ate_functional
from average_treatment_effect.lasso.RieszLasso import RieszLasso


data = DatasetHighDim.simulate_dataset(2500)
rlasso = RieszLasso(ate_functional)
rlasso.fit(data)

#out_model = OutcomeLASSO(ate_functional)
#out_model.fit(data)
#out_model.predict(data)
#print(out_model.get_plugin_estimate(data))
