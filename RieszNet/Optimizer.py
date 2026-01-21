from torch.optim import AdamW,Adam


class Optimizer:
    def __init__(self, network, epochs=1000, procedure=AdamW, early_stopping=None, lr=0.001, weight_decay=0.001):
        if early_stopping is None:
            early_stopping = {"rounds": 20, "tolerance": 0.0, "proportion": 0.8}

        optimizer_params = [
            {
                "params": [p for n, p in network.named_parameters() if n != "epsilon"],
                "weight_decay": weight_decay,
                "lr":lr
            },
            {"params": [network.epsilon], "weight_decay": 0.0, "lr":lr},
        ]

        self.optim = procedure(optimizer_params)
        self.early_stopping = early_stopping
        self.epochs = epochs

# TODO : this can be formulated in the above?
class OptimizerParams:
    def __init__(self, networks, epochs = 1000, procedure = AdamW, early_stopping = None, lr=0.001, weight_decay=0.001):
        if early_stopping is None:
            self.early_stopping = {"rounds": 20, "tolerance": 0.0, "proportion": 0.8}
        self.epochs = epochs
        params = []
        for network in networks:
            params += list(network.parameters())
        self.params = params
        self.optim = procedure(params, lr=lr, weight_decay=weight_decay)
