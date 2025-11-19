import torch
from average_treatment_effect import DragonNet
from average_treatment_effect import dataset
model = DragonNet.DragonNet()
criterion = DragonNet.DragonNetLoss()
data = dataset.Dataset.load_chernozhukov_replication(1)

covariates = data.get_as_tensor("covariates")
treatments = data.get_as_tensor("treatments")
outcomes = data.get_as_tensor("outcomes")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(1000):
    optimizer.zero_grad()

    propensity_outputs, _, _, regression_outputs = model(covariates, treatments)
    loss = criterion(propensity_outputs, treatments, regression_outputs, outcomes)
    loss.backward()
    optimizer.step()
    print(i)

_,q0,q1,_ = model(covariates, treatments)
print(torch.mean(q1-q0).item())
print(data.get_average_treatment_effect())





