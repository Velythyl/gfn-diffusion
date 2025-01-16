import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet, find_density_minmax, recenter


class _UnnormalizedDensity(BaseSet):
    def __init__(self, device, dim=2):
        super().__init__()
        self.device = device
        self.data = torch.tensor([0.0])
        self.data_ndim = 2
        self.dim = dim

        self.min_param, self.max_param = 0., 100.      # fixme hardcoded for now...

        self.MIN, self.MAX = self.get_minmax()

        self.SAMPLE_DISABLED = True


    def get_minmax(self):
        description = find_density_minmax(self, self.dim, 0, 100)
        MIN = description["min"]
        MAX = description["max"]
        return MIN.item(), MAX.item()

    def score(self, x):
        raise NotImplementedError()

    def scaled_score(self, x):
        return recenter(self.score(x).clip(self.MIN, self.MAX), self.MIN, self.MAX).clip(0.0001, 1.0)

    def unnormalized_log_prob(self, x):
        x = x.clip(self.min_param, self.max_param)
        return torch.log(torch.vmap(self.scaled_score)(x))

    def gt_logz(self):
        raise NotImplementedError()

    def energy(self, x):
        return -self.unnormalized_log_prob(x).flatten()

    def sample(self, batch_size):
        raise NotImplementedError()

    def viz_pdf(self, fsave="ou-density.png"):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()
