import torch
from torch import nn
from dope_neural_nets.dataset import Dataset
import copy


class ModelWrapper:
    def __init__(self):
        self.model = Model()

    def train_outcome_head(self, data: Dataset, train_shared_layers):
        self.model.train()
        for param in self.model.riesz_layers.parameters():
            param.requires_grad = False
        if train_shared_layers:
            for param in self.model.shared_layers.parameters():
                param.requires_grad = True
        else:
            for param in self.model.shared_layers.parameters():
                param.requires_grad = True
        train_data, val_data = data.test_train_split(0.8)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        best = 1e6
        patience = 20
        counter = 20
        best_state = None
        for epoch in range(1000):
            optimizer.zero_grad()
            predictions = self.model.predict_outcome(train_data.net_input)
            loss = criterion(predictions, train_data.outcome_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = self.model.predict_outcome(val_data.net_input)
                test_loss = criterion(predictions, val_data.outcome_tensor).item()
            if test_loss < best:
                best = test_loss
                counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            if counter >= patience:
                break
        self.model.load_state_dict(best_state)


class Model(nn.Module):
    def __init__(self, hidden_size=64):
        super(Model, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(11, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.outcome_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.riesz_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def predict_outcome(self, x):
        x = self.shared_layers(x)
        return self.outcome_layers(x)

    def predict_riesz(self, x):
        x = self.shared_layers(x)
        return self.riesz_layers(x)
