import jax
from jax2torch import jax2torch
import numpy as np
import matplotlib.pyplot as plt
from .control2d_jax import eval_actions
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

        self.SAMPLE_DISABLED = True

        #self.compiled_rollout = torch.compile(self.rollout, backend="aot_eager", mode="reduce-overhead")

        self.compiled_rollout= jax2torch(jax.jit(jax.vmap(eval_actions)))

    def gt_logz(self):
        raise NotImplementedError()

    def energy(self, x):
        return -self.control2d_log_pdf(x)

    def score_single_state(self, state):
        distance_to_goal = torch.linalg.norm(state[:2] - self.goal_pos)
        distance_reward = (self.max_distance_to_goal - distance_to_goal) / self.max_distance_to_goal

        return distance_reward + 100. * self.is_on_goal(state[:2]) #* (state[-1] > 45).float()

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
        x = torch.nan_to_num(x, nan=-torch.inf)
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

        rewards[:,:-1] = 0

        return rewards, trajectories

    def rewards_batch(self, x):
        x = x.reshape(x.shape[0], 50, 2)
        scores, _ = self.compiled_rollout(x)
        #scores = scores.sum(axis=1)
        return scores

    def score_batch(self, x):
        def reshape(_x):
            _x = _x.reshape(50,2)
            return _x
        x = torch.vmap(reshape)(x)
        scores, _ = self.compiled_rollout(x)
        scores[:,:-1] = 0
        return scores

    def control2d_log_pdf(self, x):
        scores = self.score_batch(x)
        scores = scores.sum(axis=1)
        scores = scores.clip(0.00001, 101)
        return torch.log(scores / 101) - 0.00001
    
    def sample(self, batch_size):
        raise NotImplementedError()

    def viz_pdf(self, fsave="density.png", lim=3):
        raise NotImplementedError()

    def __getitem__(self, idx):
        del idx
        return self.data[0]

    def display(self, x):
        rewards, array_states = self.rollout(x)
        poses = array_states[:, :, :2]

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Constants for the map
        MAP_WIDTH = 200
        MAP_HEIGHT = 100

        # Scaling factor to convert map coordinates to screen coordinates
        SCALE_X = 1  # Example scaling factor for the X-axis (screen width / map width)
        SCALE_Y = 1  # Example scaling factor for the Y-axis (screen height / map height)

        # Convert map coordinates to plot coordinates
        def map_to_plot(x, y):
            plot_x = x * SCALE_X
            plot_y = MAP_HEIGHT / 2 - y * SCALE_Y  # Invert Y-axis to match the coordinate system
            return plot_x, plot_y

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw the green circle at (90, 0)
        goal_x, goal_y = map_to_plot(180, 0)
        goal = patches.Circle(
            (goal_x, goal_y),
            radius=10 * SCALE_X,  # Scale the radius to match the screen size
            linewidth=2,
            edgecolor='green',
            facecolor='green'
        )
        ax.add_patch(goal)

        import matplotlib.cm as cm
        cmap = cm.get_cmap('twilight_shifted')  # You can choose any colormap
        for i, xy in enumerate(poses):
            x, y = xy[:, 0], xy[:, 1]
            x, y = torch.vmap(map_to_plot)(x, y)
            color = cmap(i / poses.shape[0])
            ax.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), color=color)

        # Set axis limits and aspect ratio
        ax.set_xlim(0, MAP_WIDTH * SCALE_X)
        ax.set_ylim(0, MAP_HEIGHT * SCALE_Y)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])

        return fig
