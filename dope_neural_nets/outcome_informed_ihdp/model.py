import torch
from torch import nn
from dope_neural_nets.dataset import Dataset
import copy
from torch.utils.data import DataLoader, TensorDataset


class ModelWrapper:
    def __init__(self, in_, hidden_size, n_shared, n_not_shared, type_="shared_base"):
        self.model = Model(in_, hidden_size, type_, n_shared, n_not_shared)

    def get_estimate_components(self, data: Dataset):
        self.model.eval()
        treated, control = data.get_counterfactual_datasets()
        outcome = self.model.predict_outcome(data.net_input)
        treated_outcome = self.model.predict_outcome(treated.net_input)
        control_outcome = self.model.predict_outcome(control.net_input)
        riesz = self.model.predict_riesz(data.net_input)
        return treated_outcome - control_outcome + riesz * (data.outcomes_tensor - outcome)

    def train_as_riesz_net(self, data: Dataset, rr_w=1, tmle_w=0, mse_w=1, lr=1e-3, wd=1e-3, patience=30):
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

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        best = 1e6
        counter = 0
        best_state = copy.deepcopy(self.model.state_dict())
        for epoch in range(1000):
            self.model.train()
            optimizer.zero_grad()
            actual_riesz = self.model.predict_riesz(train_data.net_input)
            treated_riesz = self.model.predict_riesz(train_treated.net_input)
            control_riesz = self.model.predict_riesz(train_control.net_input)
            riesz_loss = riesz_criterion(actual_riesz, treated_riesz, control_riesz)
            base_predictions = self.model.predict_without_correction(train_data.net_input)
            outcome_loss = outcome_criterion(base_predictions, train_data.outcomes_tensor)
            tmle_w_loss = tmle_criterion(self.model.predict_outcome(train_data.net_input), train_data.outcomes_tensor)
            l2_penalty = 0.0
            for name, param in self.model.named_parameters():
                if name != "epsilon":
                    l2_penalty += torch.sum(param**2)
            loss = riesz_loss * rr_w + outcome_loss * mse_w + tmle_w_loss * tmle_w + l2_penalty * wd
            loss.backward()
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                actual_riesz = self.model.predict_riesz(val_data.net_input)
                treated_riesz = self.model.predict_riesz(val_treated.net_input)
                control_riesz = self.model.predict_riesz(val_control.net_input)
                riesz_loss = riesz_criterion(actual_riesz, treated_riesz, control_riesz)
                base_predictions = self.model.predict_without_correction(val_data.net_input)
                outcome_loss = outcome_criterion(base_predictions, val_data.outcomes_tensor)
                tmle_w_loss = tmle_criterion(self.model.predict_outcome(val_data.net_input), val_data.outcomes_tensor)
                test_loss = riesz_loss * rr_w + outcome_loss * mse_w + tmle_w_loss * tmle_w
            if test_loss.item() < best:
                best = test_loss.item()
                counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                counter += 1
            if counter >= patience:
                break
        self.model.load_state_dict(best_state)
        print(epoch, self.model.epsilon)
        return best


class Model(nn.Module):
    def __init__(self, in_, hidden_size, type_, n_shared, n_not_shared):
        super(Model, self).__init__()
        if type_ == "shared_base":
            shared_layers = [HiddenLayer(in_, hidden_size)]
            for i in range(n_shared - 1):
                shared_layers.append(HiddenLayer(hidden_size, hidden_size))
            shared_layers = nn.Sequential(*shared_layers)
            self.outcome_base = shared_layers
            self.riesz_base = shared_layers
        elif type_ == "separate_nets":
            outcome_base = [HiddenLayer(in_, hidden_size)]
            for i in range(n_shared - 1):
                outcome_base.append(HiddenLayer(hidden_size, hidden_size))
            self.outcome_base = nn.Sequential(*outcome_base)
            riesz_base = [HiddenLayer(in_, hidden_size)]
            for i in range(n_shared - 1):
                riesz_base.append(HiddenLayer(hidden_size, hidden_size))
            self.riesz_base = nn.Sequential(*riesz_base)

        self.outcome_layers = BiHead(hidden_size, hidden_size, n_not_shared)
        self.riesz_layers = Head(hidden_size + 1, 100, 1)
        self.epsilon = torch.nn.Parameter(torch.tensor(0.0))

    def predict_outcome(self, x):
        return self.predict_without_correction(x) + self.epsilon * self.predict_riesz(x)

    def predict_without_correction(self, x):
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
    def forward(self, actual_riesz, treated_riesz, control_riesz):
        return torch.mean(actual_riesz**2 - 2 * (treated_riesz - control_riesz))


class BiHead(nn.Module):
    def __init__(self, in_, hidden_size, n_hidden):
        super(BiHead, self).__init__()
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

    def forward(self, x, treat):
        xt = self.t_layers(x)
        xc = self.c_layers(x)
        return treat * xt + (1 - treat) * xc


class Head(nn.Module):
    def __init__(self, in_, hidden_size, n_hidden):
        super(Head, self).__init__()
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
            # nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.layers(x)
