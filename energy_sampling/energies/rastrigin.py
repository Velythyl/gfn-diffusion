import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet, find_density_minmax, recenter
from .UnnormalizedDensity import _UnnormalizedDensity


class Rastrigin(_UnnormalizedDensity):

    @property
    def dimbounds(self, dim):
        if dim == 2:
            return None
        return -127.4579, -0.021144867

    def score(self, x):
        x = x.clip(0, 100) / 100 * 10.24
        x = x - 5.12  # recenter

        first_term = 10 * x.shape[-1]
        second_term = torch.sum((x + -5.12 * 0.4) ** 2 - 10 * torch.cos(2 * torch.pi * (x + 5.12 * 0.4)))
        return -(first_term + second_term)
