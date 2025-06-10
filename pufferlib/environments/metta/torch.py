import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pufferlib.models

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)

class Policy(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(21, cnn_channels, 5, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size//2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(21, hidden_size//2)),
            nn.ReLU(),
        )

        max_vec = torch.tensor([  1.,   9.,   1.,  30.,   1.,   3., 255.,  26.,   1.,   1.,   1.,   1.,
          1.,  47.,   3.,   3.,   2.,   1.,   1.,   1.,   1.])[None, :, None, None]
        self.register_buffer('max_vec', max_vec)

        self.actor = pufferlib.pytorch.layer_init(
	    nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

        #self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, observations, state=None):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return (actions, value), hidden

    def encode_observations(self, observations, state=None):
        features = observations.permute(0, 3, 1, 2).float() / self.max_vec
        #mmax = features.max(0)[0].max(1)[0].max(1)[0]
        #self.max_vec = torch.maximum(self.max_vec, mmax[None, :, None, None])
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden):
        #hidden = self.layer_norm(hidden)
        logits = self.actor(hidden)
        value = self.value(hidden)
        return logits, value
