"""A small convolutional front end for the board (training only; needs torch).

The observation's 12 planes go through three 3x3 convolutions (ReLU, padded, so the 9x9 grid
keeps its shape), get flattened, and the two scalars (logs left, rounds left) are appended.
The policy and value heads are ordinary tanh MLPs on top. beaver_env.Net runs the same maths
in numpy, and the JS port in the browser.
"""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from beaver_env import N_CH


class BoardCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, size: int = 9, channels: tuple[int, ...] = (32, 32, 8)):
        n2 = size * size
        super().__init__(observation_space, features_dim=channels[-1] * n2 + 2)
        self.size = size
        layers: list[nn.Module] = []
        c_in = N_CH
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU()]
            c_in = c
        self.convs = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n2 = self.size * self.size
        planes = obs[:, : N_CH * n2].reshape(-1, N_CH, self.size, self.size)
        feats = self.convs(planes).flatten(1)
        return torch.cat([feats, obs[:, N_CH * n2 :]], dim=1)


def conv_layers(model) -> list[dict]:
    """The conv weights as plain lists: [{w: out x in x 3 x 3, b: out}]."""
    ext = model.policy.pi_features_extractor
    return [
        {"w": m.weight.detach().cpu().numpy().tolist(), "b": m.bias.detach().cpu().numpy().tolist()}
        for m in ext.convs
        if isinstance(m, nn.Conv2d)
    ]
