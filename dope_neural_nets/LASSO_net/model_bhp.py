import torch
from torch import nn
from dope_neural_nets.dataset import Dataset
import copy
from torch.utils.data import DataLoader, TensorDataset


class ModelWrapper:
    def __init__(self, in_, hidden_size, n_shared, n_not_shared, type_="shared_base"):
        self.model = Model(in_, hidden_size, type_, n_shared, n_not_shared)
        self.outcome_criterion = nn.MSELoss()

    def get_estimate_components(self, data: Dataset):
        self.model.eval()
        treated, control = data.get_counterfactual_datasets()
        outcome = self.model.predict_outcome(data.net_input)
        treated_outcome = self.model.predict_outcome(treated.net_input)
        control_outcome = self.model.predict_outcome(control.net_input)
        riesz = self.model.predict_riesz(data.net_input)
        return treated_outcome - control_outcome + riesz * (data.outcomes_tensor - outcome)

    def train_outcome_head(self, data: Dataset, train_shared_layers, lr=1e-3, wd=1e-3, patience=30, l1_penalty=0):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.outcome_layers.parameters():
            param.requires_grad = True
        if train_shared_layers:
            for param in self.model.outcome_base.parameters():
                param.requires_grad = True
        train_data, val_data = data.test_train_split(0.8)
        loader = DataLoader(
            TensorDataset(train_data.net_input, train_data.outcomes_tensor), batch_size=64, shuffle=True
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=2,
            min_lr=1e-6,
        )
        best = 1e6
        counter = 0
        best_state = copy.deepcopy(self.model.state_dict())
        for epoch in range(1000):
            self.model.train()
            for x, y in loader:
                optimizer.zero_grad()
                mse_loss = self._get_mse_loss(x, y)
                l1_loss = l1_penalty * self.model.outcome_base.lasso_layer.weight.abs().sum()
                loss = mse_loss + l1_loss
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                test_loss = self._get_mse_loss(data.net_input, data.outcomes_tensor)
                scheduler.step(test_loss)
            if test_loss.item() < best:
                best = test_loss.item()
                counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break
        self.model.load_state_dict(best_state)
        return best

    def _get_mse_loss(self, x, y):
        predictions = self.model.predict_outcome(x)
        mse_loss = self.outcome_criterion(predictions, y)
        return mse_loss

    def train_riesz_head(self, data: Dataset, train_shared_layers, lr=1e-3, patience=30):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.riesz_layers.parameters():
            param.requires_grad = True
        if train_shared_layers:
            for param in self.model.riesz_base.parameters():
                param.requires_grad = True
        criterion = RieszLoss()
        train_data, val_data = data.test_train_split(0.8)
        train_treated, train_control = train_data.get_counterfactual_datasets()
        val_treated, val_control = val_data.get_counterfactual_datasets()
        loader = DataLoader(
            TensorDataset(train_data.net_input, train_treated.net_input, train_control.net_input),
            batch_size=64,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-3,
            threshold_mode="rel",
            cooldown=2,
            min_lr=1e-6,
        )
        best = 1e6
        best_state = copy.deepcopy(self.model.state_dict())
        counter = 0
        for epoch in range(1000):
            self.model.train()
            for x, xt, xc in loader:
                optimizer.zero_grad()
                actual_riesz = self.model.predict_riesz(x)
                treated_riesz = self.model.predict_riesz(xt)
                control_riesz = self.model.predict_riesz(xc)
                loss = criterion(actual_riesz, treated_riesz, control_riesz)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                actual_riesz = self.model.predict_riesz(val_data.net_input)
                treated_riesz = self.model.predict_riesz(val_treated.net_input)
                control_riesz = self.model.predict_riesz(val_control.net_input)
                test_loss = criterion(actual_riesz, treated_riesz, control_riesz)
                scheduler.step(test_loss)
            if test_loss.item() < best:
                best = test_loss.item()
                counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break
        self.model.load_state_dict(best_state)


class Model(nn.Module):
    def __init__(self, in_, hidden_size, type_, n_shared, n_not_shared):
        super(Model, self).__init__()
        shared_layers = SharedLayers(in_=in_, hidden_size=hidden_size, n_shared=n_shared)
        self.outcome_base = shared_layers
        self.riesz_base = shared_layers
        self.outcome_layers = Head(hidden_size + 1, hidden_size, n_not_shared)
        self.riesz_layers = Head(hidden_size + 1, hidden_size, n_not_shared)

    def predict_outcome(self, x):
        treat = x[:, 0].reshape(-1, 1)
        x = x[:, 1:]
        x = self.outcome_base(x)
        return self.outcome_layers(x, treat)

    def predict_riesz(self, x):
        treat = x[:, 0].reshape(-1, 1)
        x = x[:, 1:]
        x = self.riesz_base(x)
        return self.riesz_layers(x, treat)


class RieszLoss(nn.Module):
    def forward(self, grad, prediction):
        return torch.mean(prediction**2 - 2 * grad)


class SharedLayers(nn.Module):
    def __init__(self, in_, hidden_size, n_shared):
        super(SharedLayers, self).__init__()
        shared_layers = [HiddenLayer(in_, hidden_size)]
        for i in range(n_shared - 1):
            shared_layers.append(HiddenLayer(hidden_size, hidden_size))
        self.shared_layers = nn.Sequential(*shared_layers)
        self.lasso_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.shared_layers(x)
        return self.lasso_layer(x)


class BiHead(nn.Module):
    def __init__(self, in_, hidden_size, n_hidden):
        super(BiHead, self).__init__()
        if n_hidden > 0:
            t = [HiddenLayer(in_, hidden_size)]
            for i in range(n_hidden - 1):
                t.append(HiddenLayer(hidden_size, hidden_size))
            t.append(nn.Linear(hidden_size, 1))
            self.t_layers = nn.Sequential(*t)

            c = [HiddenLayer(in_, hidden_size)]
            for i in range(n_hidden - 1):
                c.append(HiddenLayer(hidden_size, hidden_size))
            c.append(nn.Linear(hidden_size, 1))
            self.c_layers = nn.Sequential(*c)
        else:
            self.t_layers = nn.Linear(in_, 1)
            self.c_layers = nn.Linear(in_, 1)

    def forward(self, x, treat):
        xt = self.t_layers(x)
        xc = self.c_layers(x)
        return treat * xt + (1 - treat) * xc


class Head(nn.Module):
    def __init__(self, in_, hidden_size, n_hidden):
        super(Head, self).__init__()
        if n_hidden > 0:
            layers = [HiddenLayer(in_, hidden_size)]
            for i in range(n_hidden - 1):
                layers.append(HiddenLayer(hidden_size, hidden_size))
            layers.append(nn.Linear(hidden_size, 1))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Linear(in_, 1)

    def forward(self, x, treat):
        x = torch.cat((x, treat), dim=1)
        return self.layers(x)


class HiddenLayer(nn.Module):
    def __init__(self, in_, out_):
        super(HiddenLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_, out_),
            nn.ELU(),
        )

    def forward(self, x):
        return self.layers(x)
