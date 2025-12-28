from torch.optim import AdamW


class Optimizer:
    def __init__(self, network, epochs=1000, procedure=AdamW, early_stopping=None, lr=0.0001, weight_decay=0.01):
        if early_stopping is None:
            early_stopping = {"rounds": 50, "tolerance": 0.001, "proportion": 0.9}

        optimizer_params = [
            {
                "params": [p for n, p in network.named_parameters() if n != "epsilon"],
                "weight_decay": weight_decay,
            },
            {"params": [network.epsilon], "weight_decay": 0.0},
        ]

        self.optim = procedure(optimizer_params, lr=lr)
        self.early_stopping = early_stopping
        self.epochs = epochs
