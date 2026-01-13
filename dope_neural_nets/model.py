import torch
from torch import nn
from dope_neural_nets.dataset import Dataset
import copy


class ModelWrapper:
    def __init__(self, type="shared_base"):
        self.model = Model(type=type)

    def get_estimate_components(self, data: Dataset):
        self.model.eval()
        treated, control = data.get_counterfactual_datasets()
        outcome = self.model.predict_outcome(data.net_input)
        treated_outcome = self.model.predict_outcome(treated.net_input)
        control_outcome = self.model.predict_outcome(control.net_input)
        riesz = self.model.predict_riesz(data.net_input)
        riesz = torch.clamp(riesz, -100, 100)
        return treated_outcome - control_outcome + riesz * (data.outcomes_tensor - outcome)

    def train_as_riesz_net(self, data: Dataset, rr_w=1, tmle_w=0, mse_w=1):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        if tmle_w == 0:
            self.model.epsilon.requires_grad = False
        riesz_criterion = RieszLoss()
        outcome_criterion = nn.MSELoss()
        tmle_criterion = nn.MSELoss()
        train_data, val_data = data.test_train_split(0.8)
        train_treated, train_control = train_data.get_counterfactual_datasets()
        val_treated, val_control = val_data.get_counterfactual_datasets()
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
            riesz_loss = riesz_criterion(actual_riesz, treated_riesz, control_riesz)
            base_predictions = self.model.predict_without_correction(train_data.net_input)
            outcome_loss = outcome_criterion(base_predictions, train_data.outcomes_tensor)
            tmle_w_loss = tmle_criterion(self.model.predict_outcome(train_data.net_input), train_data.outcomes_tensor)
            loss = riesz_loss * rr_w + outcome_loss * mse_w + tmle_w_loss * tmle_w
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                actual_riesz = self.model.predict_riesz(val_data.net_input)
                treated_riesz = self.model.predict_riesz(val_treated.net_input)
                control_riesz = self.model.predict_riesz(val_control.net_input)
                riesz_loss = riesz_criterion(actual_riesz, treated_riesz, control_riesz)
                base_predictions = self.model.predict_without_correction(val_data.net_input)
                outcome_loss = outcome_criterion(base_predictions, val_data.outcomes_tensor)
                tmle_w_loss = tmle_criterion(self.model.predict_outcome(val_data.net_input), val_data.outcomes_tensor)
                test_loss = (riesz_loss * rr_w + outcome_loss * mse_w + tmle_w_loss * tmle_w).item()
            if test_loss < best:
                best = test_loss
                counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break
        print(outcome_loss, riesz_loss,test_loss)
        self.model.load_state_dict(best_state)

    def train_outcome_head(self, data: Dataset, train_shared_layers):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.outcome_layers.parameters():
            param.requires_grad = True
        if train_shared_layers:
            for param in self.model.outcome_base.parameters():
                param.requires_grad = True
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=1e-3,
        )

        train_data, val_data = data.test_train_split(0.8)
        best = 1e6
        patience = 20
        counter = 0
        best_state = None
        for epoch in range(1000):
            optimizer.zero_grad()
            predictions = self.model.predict_without_correction(train_data.net_input)
            loss = criterion(predictions, train_data.outcomes_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = self.model.predict_without_correction(val_data.net_input)
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
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.riesz_layers.parameters():
            param.requires_grad = True
        if train_shared_layers:
            for param in self.model.riesz_base.parameters():
                param.requires_grad = True
        criterion = RieszLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3,
            weight_decay=1e-3,
        )

        train_data, val_data = data.test_train_split(0.8)
        train_treated, train_control = train_data.get_counterfactual_datasets()
        val_treated, val_control = val_data.get_counterfactual_datasets()
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
    def __init__(self, hidden_size=64, type="shared_base"):
        super(Model, self).__init__()
        if type == "shared_base":
            shared_layers = nn.Sequential(
                nn.Linear(11, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.outcome_base = shared_layers
            self.riesz_base = shared_layers
        elif type == "separate_nets":
            self.outcome_base = nn.Sequential(
                nn.Linear(11, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.riesz_base = nn.Sequential(
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
        self.epsilon = torch.nn.Parameter(torch.tensor(0.0))

    def predict_outcome(self, x):
        return self.outcome_layers(self.outcome_base(x)) + self.epsilon * self.predict_riesz(x)

    def predict_without_correction(self, x):
        return self.outcome_layers(self.outcome_base(x))

    def predict_riesz(self, x):
        x = self.riesz_base(x)
        return self.riesz_layers(x)


class RieszLoss(nn.Module):
    def forward(self, actual_riesz, treated_riesz, control_riesz):
        return torch.mean(actual_riesz**2 - 2 * (treated_riesz - control_riesz))
