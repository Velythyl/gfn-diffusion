import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet


class FiveGaussianMixture(BaseSet):
    def __init__(self, device, dim=2):
        super().__init__()
        self.device = device
        self.data = torch.tensor([0.0])
        self.data_ndim = 2

        mean = torch.rand((5,dim), device=self.device) * 100   # support is 0-100

        nmode = len(mean)
        comp = D.Independent(D.Normal(mean.to(self.device), torch.ones_like(mean).to(self.device)), 1)
        mix = D.Categorical(torch.ones(nmode).to(self.device))
        self.gmm = MixtureSameFamily(mix, comp)
        self.data_ndim = dim
        self.means = mean

    @property
    def params(self):
        return torch.clone(self.means)

    def gt_logz(self):
        return 0.

    def energy(self, x):
        return -self.gmm.log_prob(x).flatten()

    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))

    def viz_pdf(self, fsave="ou-density.png"):
        x = torch.linspace(-8, 8, 100).to(self.device)
        y = torch.linspace(-8, 8, 100).to(self.device)
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?

        density = self.unnorm_pdf(x)
        return x, density

    def __getitem__(self, idx):
        del idx
        return self.data[0]
