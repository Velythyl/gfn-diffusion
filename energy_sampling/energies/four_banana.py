import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet


class FourBananaMixture(BaseSet):
    def __init__(self, device, dim=2):
        super().__init__()
        assert dim == 2
        self.device = device
        self.data = torch.tensor([0.0])
        self.data_ndim = 2
        self.dim = dim

        self.locs = torch.rand((4,dim), device=self.device) * 100   # support is 0-100

        self.SAMPLE_DISABLED = True

    @property
    def params(self):
        return torch.clone(self.locs)

    def one_banana(self, loc, x):
        x = x.clip(0, 100)  # 0-100 support
        x = x - loc
        x1, x2 = x[0], x[1]
        return -0.5 * (0.03 * x1 * x1 + (x2 + 0.03 * (x1 * x1 - 100))**2)

    def four_banana(self, x):
        ret = self.one_banana(self.locs[0], x)
        for i in range(1, 4, 1):
            ret = torch.logaddexp(ret, self.one_banana(i, x))
        return ret

    def log_prob(self,x):
        return torch.vmap(self.four_banana)(x)

    def gt_logz(self):
        raise NotImplementedError()

    def energy(self, x):
        x = x + 50  # maps -50,50 to 0,100 because GFNDiffusion has a hard time with exploring to 100
        return -self.log_prob(x).flatten()

    def sample(self, batch_size):
        raise NotImplementedError()

    def viz_pdf(self, fsave="ou-density.png"):
        x = torch.linspace(0, 100, 100).to(self.device)
        y = torch.linspace(0, 100, 100).to(self.device)
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?

        density = self.unnorm_pdf(x)
        return x, density

    def __getitem__(self, idx):
        del idx
        return self.data[0]
