import torch
from torch import nn
from dope_neural_nets.dataset import Dataset
import copy


class ModelWrapper:
    def __init__(self):
        self.model = Model()

    def get_estimate_components(self, data: Dataset):
        self.model.eval()
        treated, control = data.get_counterfactual_datasets()
        outcome = self.model.predict_outcome(data.net_input)
        treated_outcome = self.model.predict_outcome(treated.net_input)
        control_outcome = self.model.predict_outcome(control.net_input)
        riesz = self.model.predict_riesz(data.net_input)
        return treated_outcome - control_outcome + riesz * (data.outcomes_tensor - outcome)

    def train_outcome_head(self, data: Dataset, train_shared_layers):
        self.model.train()
        for param in self.model.outcome_layers.parameters():
            param.requires_grad = True
        for param in self.model.riesz_layers.parameters():
            param.requires_grad = False
        if train_shared_layers:
            for param in self.model.shared_layers.parameters():
                param.requires_grad = True
        else:
            for param in self.model.shared_layers.parameters():
                param.requires_grad = False
        train_data, val_data = data.test_train_split(0.8)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=1e-3,
        )
        best = 1e6
        patience = 20
        counter = 0
        best_state = None
        for epoch in range(1000):
            optimizer.zero_grad()
            predictions = self.model.predict_outcome(train_data.net_input)
            loss = criterion(predictions, train_data.outcomes_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = self.model.predict_outcome(val_data.net_input)
                test_loss = criterion(predictions, val_data.outcomes_tensor).item()
            if test_loss < best:
                best = test_loss
                counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break
        self.model.load_state_dict(best_state)

    def train_riesz_head(self, data: Dataset, train_shared_layers):
        self.model.train()
        for param in self.model.outcome_layers.parameters():
            param.requires_grad = False
        for param in self.model.riesz_layers.parameters():
            param.requires_grad = True
        if train_shared_layers:
            for param in self.model.shared_layers.parameters():
                param.requires_grad = True
        else:
            for param in self.model.shared_layers.parameters():
                param.requires_grad = False
        train_data, val_data = data.test_train_split(0.8)
        train_treated, train_control = train_data.get_counterfactual_datasets()
        val_treated, val_control = val_data.get_counterfactual_datasets()
        criterion = RieszLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=1e-3,
        )
        best = 1e6
        patience = 20
        counter = 0
        best_state = None
        for epoch in range(1000):
            optimizer.zero_grad()
            actual_riesz = self.model.predict_riesz(train_data.net_input)
            treated_riesz = self.model.predict_riesz(train_treated.net_input)
            control_riesz = self.model.predict_riesz(train_control.net_input)
            loss = criterion(actual_riesz, treated_riesz, control_riesz)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                actual_riesz = self.model.predict_riesz(val_data.net_input)
                treated_riesz = self.model.predict_riesz(val_treated.net_input)
                control_riesz = self.model.predict_riesz(val_control.net_input)
                test_loss = criterion(actual_riesz, treated_riesz, control_riesz).item()
            if test_loss < best:
                best = test_loss
                counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
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


class RieszLoss(nn.Module):
    def forward(self, actual_riesz, treated_riesz, control_riesz):
        return torch.mean(actual_riesz**2 - 2 * (treated_riesz - control_riesz))
