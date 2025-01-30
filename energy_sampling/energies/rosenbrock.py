import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet, find_density_minmax, recenter
from .UnnormalizedDensity import _UnnormalizedDensity


class Rosenbrock(_UnnormalizedDensity):
    @property
    def dimbounds(self, dim):
        if dim == 2:
            MIN = 37.251522
            MAX = 42367.33
        else:
            MIN = 10
            MAX = 42367.33 * 2
        return MIN, MAX

    def __init__(self,device, dim=2, m=10):
        super().__init__(device, dim)
        self.m = m

    def score(self, x):
        x = x / self.max_param * 15 - 5  # -5, 10
        x = x.clip(-5, 10)

        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0, axis=0)
