import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet, find_density_minmax, recenter
from .UnnormalizedDensity import _UnnormalizedDensity


class Michalewicz(_UnnormalizedDensity):
    def __init__(self,device, dim=2, m=10):
        self.m = m
        super().__init__(device, dim)

    @property
    def dimbounds(self, dim):
        temp = {
            10: [-6.3356304, -1.1230314e-16]
        }
        if dim in temp:
            return temp[dim]
        return None

    def score(self, x):
        x = x /self.max_param * torch.pi
        x = x.clip(0, torch.pi)

        def thunk(d, i):
            return torch.sin(i) * torch.sin((d * i ** 2) / torch.pi) ** (2 * self.m)

        ret = torch.vmap(thunk)(1 + torch.arange(self.dim, device=self.device), x)
        return -ret.sum()
