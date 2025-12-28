import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RieszNetModule:
    def __init__(self, network, optimizer, loss):
        self.network = network
        self.optimizer = optimizer
        self.loss = loss

    def fit(self, data):
        pass

    def get_plugin(self, data):
        return self.network.get_plugin_estimate(data)

    def get_correction(self, data):
        residuals = self.network.get_residuals(data)
        rr = self.network.get_riesz_representer(data)
        return residuals * rr

    def get_functional(self, data):
        return self.network.get_functional(data)

    def get_double_robust(self, data):
        plugin = self.get_plugin(data)
        correction = self.get_correction(data)
        return plugin + np.mean(correction)

