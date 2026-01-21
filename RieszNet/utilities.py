import torch.nn as nn
def make_sequential(in_dim,hidden_dim,out_dim,n_hidden,activation=nn.ELU, activate_all=False):
    layers = []
    dims = [in_dim] + [hidden_dim] * n_hidden + [out_dim]

    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(d_in, d_out))
        if d_out != out_dim or activate_all:
            layers.append(activation())

    return nn.Sequential(*layers)