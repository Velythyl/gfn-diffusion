import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_set import BaseSet


class Control2D(BaseSet):
    """
    x0 ~ N(0, 3^2), xi | x0 ~ N(0, exp(x0)), i = 1, ..., 9
    """
    def __init__(self, device, dim=100):
        super().__init__()
        # xlim = 0.01 if nmode == 1 else xlim
        self.device = device

        self.data = torch.zeros((100,), dtype=float).to(self.device) #torch.ones(dim, dtype=float).to(self.device)
        self.data_ndim = dim

        self.goal_pos = torch.tensor([180,0], dtype=float).to(self.device)
        self.goal_size = 10.

        self.y_bounds = (50.,-50.)
        self.x_bounds = (0., 200.)
        self.max_distance_to_goal = torch.linalg.norm(torch.tensor([0, 50]).to(self.device) - self.goal_pos)
        self.extremum_positional_velocity = 5.0
        self.num_timesteps = 50
        self.action_dim = 2

    def gt_logz(self):
        raise NotImplementedError()

    def energy(self, x):
        return -self.control2d_log_pdf(x)

    def score_single_state(self, state):
        distance_to_goal = torch.linalg.norm(state[:2] - self.goal_pos)
        distance_reward = (self.max_distance_to_goal - distance_to_goal) / self.max_distance_to_goal

        return distance_reward + 100. * self.is_on_goal(state[:2]) * (state[-1] > 45).float()

    def is_on_goal(self, xy):
        distance_to_goal = torch.linalg.norm(xy - self.goal_pos)
        is_within_goal = distance_to_goal < self.goal_size
        return is_within_goal.float()

    def dynamics(self, state, action):
        position_vel_delta, heading_delta = action

        # Update velocities
        position_velocity = torch.clip(state[2] + position_vel_delta, 0,self.extremum_positional_velocity)

        # Update heading (angular displacement)
        heading = torch.remainder(state[3] + heading_delta, 2 * torch.pi)

        # Update position based on current heading and velocity
        x = state[0] + position_velocity * torch.cos(heading)
        y = state[1] + position_velocity * torch.sin(heading)

        x = torch.clip(x, 0, 200)
        y = torch.clip(y, -50, 50)

        _is_on_goal = self.is_on_goal(state[:2])
        new_x = x * (1 - _is_on_goal) + self.goal_pos[0] * _is_on_goal
        new_y = y * (1 - _is_on_goal) + self.goal_pos[1] * _is_on_goal

        ret = torch.zeros_like(state)
        ret[0] = new_x
        ret[1] = new_y
        ret[2] = position_velocity
        ret[3] = heading
        ret[4] = state[-1]+1
        return ret

    def rollout(self, x):
        x = x.reshape(x.shape[0], self.num_timesteps, self.action_dim)
        x = torch.clip(x, -1, 1)
        x[:, :, 1] = x[:, :, 1] / 10

        def initial_state(_):
            return torch.tensor([1., 0.,0.,0.,0])

        initial_states = torch.vmap(initial_state)(torch.arange(x.shape[0]))
        state = initial_states.to(self.device)

        rewards = torch.zeros((x.shape[0], self.num_timesteps), device=self.device)
        trajectories = torch.zeros((x.shape[0], self.num_timesteps, 5), device=self.device)
        for timestep in range(self.num_timesteps):
            state = torch.vmap(self.dynamics)(state, x[:,timestep])
            reward_of_state = torch.vmap(self.score_single_state)(state)
            rewards[:, timestep] = reward_of_state
            trajectories[:, timestep] = state

        return rewards, trajectories

    def score_batch(self, x):
        scores, _ = self.rollout(x)
        scores = scores.sum(axis=1)
        return scores

    def control2d_log_pdf(self, x):
        return torch.log(self.score_batch(x) / (5.5 * 101))
    
    def sample(self, batch_size):
        raise NotImplementedError()

    def viz_pdf(self, fsave="density.png", lim=3):
        raise NotImplementedError()

    def __getitem__(self, idx):
        del idx
        return self.data[0]
