import abc
import torch
import numpy as np
from torch.utils.data import Dataset


def nll_unit_gaussian(data, sigma=1.0):
    data = data.view(data.shape[0], -1)
    loss = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * data * data / (sigma ** 2)
    return torch.sum(torch.flatten(loss, start_dim=1), -1)

import torch

def find_density_minmax(density, dim, min_param, max_param):
    ALL_SCORES = torch.zeros((100, 1_000_000))

    for i in range(100):
        if i == 0 and dim == 2:
            tX = torch.arange(0, 10, 0.01, device=density.device) * max_param / 10
            tY = torch.arange(0, 10, 0.01, device=density.device) * max_param / 10
            tX, tY = torch.meshgrid(tX, tY)
            tX = torch.stack((tX, tY), axis=-1).reshape((tX.shape[0] * tY.shape[1], 2))
        else:
            tX = torch.rand((1_000_000, dim), device=density.device) * max_param
        tX = tX.clip(min_param, max_param)
        tZ = torch.vmap(density.score)(tX)

        ALL_SCORES[i] = tZ

    ALL_SCORES = ALL_SCORES.flatten()

    ret = {
        "min": ALL_SCORES.min(),
        #"1%": torch.quantile(ALL_SCORES, 0.01),
        #"10%": torch.quantile(ALL_SCORES, 0.1),
        #"50%": torch.quantile(ALL_SCORES, 0.5),
        #"90%": torch.quantile(ALL_SCORES, 0.9),
        "max": ALL_SCORES.max()
    }

    return ret

def recenter(particle, min_param, max_param):
    # recenter to [0,1] because it empirically gives nice RND values
    particle = (particle - min_param) / (max_param - min_param)
    return particle


class BaseSet(abc.ABC, Dataset):
    def __init__(self, len_data=-2333):
        self.num_sample = len_data
        self.data = None
        self.data_ndim = None
        self._gt_ksd = None

    def gt_logz(self):
        raise NotImplementedError

    @abc.abstractmethod
    def energy(self, x):
        return

    def unnorm_pdf(self, x):
        return torch.exp(-self.energy(x))

    # hmt stands for hamiltonian
    def hmt_energy(self, x):
        dim = x.shape[-1]
        x, v = torch.split(x, dim // 2, dim=-1)
        neg_log_p_x = self.sample_energy_fn(x)
        neg_log_p_v = nll_unit_gaussian(v)
        return neg_log_p_x + neg_log_p_v

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError

    def score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def log_reward(self, x):
        return -self.energy(x)

    def hmt_score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.hmt_energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data
