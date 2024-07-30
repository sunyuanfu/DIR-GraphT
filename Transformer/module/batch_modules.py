import torch
import torch.nn as nn

class Apply(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, G_features):
        G_features = self.module(G_features)
        return G_features

class Add(nn.Module):
    def forward(self, G_features, h_features):
        G_features = G_features + h_features
        return G_features