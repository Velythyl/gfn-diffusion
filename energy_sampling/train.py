import pathlib
import sys

import jax.random

from plot_utils import *
import argparse
import torch
import os

from utils import (set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name, uniform_discretizer, random_discretizer,
                   low_discrepancy_discretizer, low_discrepancy_discretizer2, shifted_equidistant)
from buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from gflownet_losses import *
from energies import *
from evaluations import *

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--buffer_size', type=int, default=300 * 1000 * 2)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='control2d_actions',
                    choices=('9gmm', '25gmm', 'hard_funnel', 'easy_funnel', 'many_well', 'control2d_actions', 'five_mvn', 'four_bananas', 'michalewicz', 'rosenbrock', 'rastrigin', 'log_cox', 'cancer', 'credit'))
parser.add_argument("--energy_dim", type=int, default=None)
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--both_ways', action='store_true', default=False)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

parser.add_argument('--ld_schedule', action='store_true', default=False)

# target acceptance rate
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--discretizer', type=str, default="random",
                    choices=('random', 'uniform', 'low_discrepancy', 'low_discrepancy2', 'equidistant', 'adaptive'))
parser.add_argument('--discretizer_max_ratio', type=float, default=10.0)
parser.add_argument('--discretizer_traj_length', type=int, default=100)
parser.add_argument('--traj_length_strategy', type=str, default="static", choices=('static', 'dynamic'))
parser.add_argument('--min_traj_length', type=int, default=10)
parser.add_argument('--max_traj_length', type=int, default=200)
parser.add_argument('--use_prior', action='store_true', default=False)
parser.add_argument('--prior_scale', type=float, default=10.0)
parser.add_argument('--wandb_name', type=str, default="GFN Energy")
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 2000
final_eval_data_size = 2000
# eval_data_size = 10000
# final_eval_data_size = 10000
if args.energy == 'cancer':
    eval_data_size = 10080
    final_eval_data_size = 10080
if args.energy == 'credit':
    eval_data_size = 10000
    final_eval_data_size = 10000
plot_data_size = 2000
final_plot_data_size = 2000

_LIST_OF_NO_SAMPLES_ENERGIES = ['log_cox']

if args.pis_architectures:
    args.zero_init = True

device = "cpu" #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.discretizer_traj_length).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True


def get_energy():
    dim = args.energy_dim

    if args.energy == '9gmm':
        energy = NineGaussianMixture(device=device)
    elif args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == 'hard_funnel':
        energy = HardFunnel(device=device)
    elif args.energy == 'easy_funnel':
        energy = EasyFunnel(device=device)
    elif args.energy == 'many_well':
        energy = ManyWell(device=device)
    elif args.energy == 'control2d_actions':
        energy = Control2D(device=device)
    elif args.energy == 'five_mvn':
        energy = FiveGaussianMixture(device=device) if dim is None else FiveGaussianMixture(device=device, dim=dim)
    elif args.energy == 'four_bananas':
        energy = FourBananaMixture(device=device)
    elif args.energy == 'michalewicz':
        energy = Michalewicz(device=device) if dim is None else Michalewicz(device=device, dim=dim)
    elif args.energy == 'rastrigin':
        energy = Rastrigin(device=device)  if dim is None else  Rastrigin(device=device, dim=dim)
    elif args.energy == 'rosenbrock':
        energy = Rosenbrock(device=device)  if dim is None else  Rosenbrock(device=device, dim=dim)
    elif args.energy == 'log_cox':
        energy = CoxDist(device=device)
    elif args.energy == 'cancer':
        energy = BreastCancer(use_prior=args.use_prior, prior_scale=args.prior_scale, device=device)
    elif args.energy == 'credit':
        energy = GermanCredit(use_prior=args.use_prior, prior_scale=args.prior_scale, device=device)
    return energy


def plot_step(energy, gfn_model, name):
    import matplotlib.pyplot as plt
    plt.close("all")

    if args.energy == 'many_well':
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward)

        vizualizations = viz_many_well(energy, samples)
        fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23 = vizualizations

        fig_samples_x13.savefig(f'{name}samplesx13.pdf', bbox_inches='tight')
        fig_samples_x23.savefig(f'{name}samplesx23.pdf', bbox_inches='tight')

        fig_kde_x13.savefig(f'{name}kdex13.pdf', bbox_inches='tight')
        fig_kde_x23.savefig(f'{name}kdex23.pdf', bbox_inches='tight')

        fig_contour_x13.savefig(f'{name}contourx13.pdf', bbox_inches='tight')
        fig_contour_x23.savefig(f'{name}contourx23.pdf', bbox_inches='tight')

        return {"visualization/contourx13": wandb.Image(fig_to_image(fig_contour_x13)),
                "visualization/contourx23": wandb.Image(fig_to_image(fig_contour_x23)),
                "visualization/kdex13": wandb.Image(fig_to_image(fig_kde_x13)),
                "visualization/kdex23": wandb.Image(fig_to_image(fig_kde_x23)),
                "visualization/samplesx13": wandb.Image(fig_to_image(fig_samples_x13)),
                "visualization/samplesx23": wandb.Image(fig_to_image(fig_samples_x23))}
    elif args.energy == 'control2d_actions':
        samples = gfn_model.sample(10000, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward)
        fig = energy.display(samples)
        return {"visualization/control2d_actions": wandb.Image(fig_to_image(fig))}

    elif energy.data_ndim != 2:
        return {}

    elif not hasattr(energy, "SAMPLE_DISABLED"):
        batch_size = plot_data_size
        samples = gfn_model.sample(batch_size, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward)
        gt_samples = energy.sample(batch_size)

        fig_contour, ax_contour = get_figure(bounds=(-13., 13.))
        fig_kde, ax_kde = get_figure(bounds=(-13., 13.))
        fig_kde_overlay, ax_kde_overlay = get_figure(bounds=(-13., 13.))

        plot_contours(energy.log_reward, ax=ax_contour, bounds=(-13., 13.), n_contour_levels=150, device=device)
        plot_kde(gt_samples, ax=ax_kde_overlay, bounds=(-13., 13.))
        plot_kde(samples, ax=ax_kde, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_contour, bounds=(-13., 13.))
        plot_samples(samples, ax=ax_kde_overlay, bounds=(-13., 13.))

        fig_contour.savefig(f'{name}contour.pdf', bbox_inches='tight')
        fig_kde_overlay.savefig(f'{name}kde_overlay.pdf', bbox_inches='tight')
        fig_kde.savefig(f'{name}kde.pdf', bbox_inches='tight')
        # return None
        return {"visualization/contour": wandb.Image(fig_to_image(fig_contour)),
                "visualization/kde_overlay": wandb.Image(fig_to_image(fig_kde_overlay)),
                "visualization/kde": wandb.Image(fig_to_image(fig_kde))}
    else:
        return {}


def eval_step(eval_data, energy, gfn_model, final_eval=False):
    gfn_model.eval()
    metrics = dict()
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics['final_eval/log_Z'], metrics['final_eval/log_Z_lb'], metrics[
            'final_eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
            'eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward)
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    else:
        if final_eval:
            metrics['final_eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                              gfn_model,
                                                                                                              lambda bsz: uniform_discretizer(bsz, args.T),
                                                                                                              energy.log_reward)
        else:
            metrics['eval/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
                                                                                                        gfn_model,
                                                                                                        lambda bsz: uniform_discretizer(bsz, args.T),
                                                                                                        energy.log_reward)
        metrics.update(get_sample_metrics(samples, eval_data, final_eval))
    gfn_model.train()
    return metrics

def eval_step_K_step_discretizer(energy, gfn_model, final_eval=False, traj_length=None):
    gfn_model.eval()
    metrics = dict()
    if traj_length is None:
        traj_length = args.discretizer_traj_length
    if final_eval:
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, metrics[f'final_eval_{traj_length}_steps/log_Z'], metrics[f'final_eval_{traj_length}_steps/log_Z_lb'], _ \
            = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, traj_length), energy.log_reward)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics[f'eval_{traj_length}_steps/log_Z'], metrics[f'eval_{traj_length}_steps/log_Z_lb'], _ \
            = log_partition_function(
            init_state, gfn_model, lambda bsz: uniform_discretizer(bsz, traj_length), energy.log_reward)
    # if eval_data is None:
    #     log_elbo = None
    #     sample_based_metrics = None
    # else:
    #     if final_eval:
    #         metrics[f'final_eval_{args.discretizer_traj_length}_steps/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
    #                                                                                                           gfn_model,
    #                                                                                                           lambda bsz: uniform_discretizer(bsz, traj_length),
    #                                                                                                           energy.log_reward)
    #     else:
    #         metrics[f'eval_{args.discretizer_traj_length}_steps/mean_log_likelihood'] = 0. if args.mode_fwd == 'pis' else mean_log_likelihood(eval_data,
    #                                                                                                     gfn_model,
    #                                                                                                     lambda bsz: uniform_discretizer(bsz, traj_length),
    #                                                                                                     energy.log_reward)
    #     metrics.update(get_sample_metrics(samples, eval_data, final_eval, K=traj_length))
    gfn_model.train()
    return metrics


def train_step(energy, gfn_model, gfn_optimizer, it, exploratory, buffer, buffer_ls, exploration_factor, exploration_wd):
    gfn_model.zero_grad()

    # discretizer = lambda bsz: uniform_discretizer(bsz, args.T)
    # discretizer = lambda bsz: uniform_discretizer(bsz, np.random.randint(10,args.T+1))
    # discretizer = lambda bsz: random_discretizer(bsz, args.T, 10)
    traj_length = args.discretizer_traj_length if args.traj_length_strategy == 'static' \
        else np.random.randint(low=args.min_traj_length, high=args.max_traj_length+1)
    if args.discretizer == 'random':
        discretizer = lambda bsz: random_discretizer(bsz, traj_length, max_ratio=args.discretizer_max_ratio)
    elif args.discretizer == 'low_discrepancy':
        discretizer = lambda bsz: low_discrepancy_discretizer(bsz, traj_length)
    elif args.discretizer == 'low_discrepancy2':
        discretizer = lambda bsz: low_discrepancy_discretizer2(bsz, traj_length)
    elif args.discretizer == 'equidistant':
        discretizer = lambda bsz: shifted_equidistant(bsz, traj_length)
    else:
        discretizer = lambda bsz: uniform_discretizer(bsz, traj_length)
    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r  = fwd_train_step(energy, gfn_model, discretizer, exploration_std, return_exp=True)
                buffer.add(states[:, -1],log_r)
            else:
                loss = fwd_train_step(energy, gfn_model, discretizer, exploration_std)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, discretizer, exploration_std, it=it)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, discretizer, exploration_std, it=it)
    else:
        loss = fwd_train_step(energy, gfn_model, discretizer, exploration_std)

    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(energy, gfn_model, discretizer, exploration_std, return_exp=False):
    init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix, discretizer,
                                exploration_std=exploration_std, return_exp=return_exp)
    return loss


def bwd_train_step(energy, gfn_model, buffer, buffer_ls, discretizer, exploration_std=None, it=0):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.batch_size, exploration_std).to(device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(device)
    elif args.sampling == 'buffer':
        if args.local_search:
            if it % args.ls_cycle < 2:
                samples, rewards = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer_ls.add(local_search_samples, log_r)
        
            samples, rewards = buffer_ls.sample()
        else:
            samples, rewards = buffer.sample()

    loss = get_gfn_backward_loss(args.mode_bwd, samples, gfn_model, energy.log_reward, discretizer,
                                 exploration_std=exploration_std)
    return loss

#sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
#sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/tpdist")
#sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/tpdist/utils")
#print(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
#exit()
#sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve())+"/")
def get_jax_eval(args, energy):
    from tpdist.evalmetrics.metrics_suite import get_eval_metrics
    from tpdist.machinery.bounds import Bounds
    filter_bounds = Bounds(min=0, max=100., dim=energy.dim)
    metrics_from_particles, metrics_from_logprob_model = get_eval_metrics(filter_bounds.expand(),
                                                                          num_samples_to_use=100_000 // 2)  # EvalMetrics_Manager(key, filter_bounds.expand(), num_samples_to_use=params.num_eval_points_to_use)
    key = torch.randint(0, 1000, size=(1,)).item()

    # get equivalent dist in jax
    from omegaconf import omegaconf
    energy_name_jax = {"control2d_actions": "control2d_actions", "four_banana": "four_bananas"}.get(args.energy, args.energy)
    cfg = omegaconf.OmegaConf.load(f"../../tpdist/config/dist/{energy_name_jax}.yaml")
    if args.energy not in ["cancer", "credit"]:
        cfg["dim"] = energy.dim

    def get_jax_key():
        return jax.random.PRNGKey(torch.randint(0, 5000, size=(1,)).item())

    from tpdist.machinery.tp_dist import resolve_tpdist
    tpdist = resolve_tpdist(cfg.distfam, filter_bounds, cfg).teleport(get_jax_key())

    def set_tpdist_params():
        dist_params = tpdist.dist_params
        from torch2jax import j2t, t2j
        new_params = []
        for i, param in enumerate(dist_params):
            param = param.replace(state=t2j(energy.params[i].cpu()))
            new_params.append(param)
        return tpdist.replace(dist_params=new_params)

    if args.energy in ["five_mvn", "four_bananas"]:
        tpdist = set_tpdist_params()


    def jax_eval(gfn_model):
        gfn_model.eval()
        from torch2jax import j2t, t2j
        samples = t2j(gfn_model.sample(10000, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward).cpu())
        gfn_model.train()

        if args.energy in ["five_mvn", "four_bananas", "rastrigin", "michalewicz", "rosenbrock", "cancer"]:
            samples = samples + 50  # maps -50,50 to 0,100 because GFNDiffusion has a hard time with exploring to 100
        elif args.energy in ["credit"]:
            samples = samples + 8   # gets removed by jax code later

        evalled = metrics_from_particles(get_jax_key(), tpdist, samples)

        CSV_BUILDER.log(evalled)
        return evalled
    return jax_eval

jax_eval = None
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve())+"/tpdist")
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve())+"/tpdist/utils")
from tpdist.utils.wandbcsv import init_no_wandb as WANDBCSV_INIT
CSV_BUILDER = WANDBCSV_INIT()


def train():
    global jax_eval

    name = get_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    energy = get_energy()
    if not hasattr(energy, "SAMPLE_DISABLED"):
        eval_data = energy.sample(eval_data_size).to(device) if not (args.energy in _LIST_OF_NO_SAMPLES_ENERGIES) else None

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project=args.wandb_name, config=config, name=name)

    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)


    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)

    print(gfn_model)
    metrics = dict()

    buffer = ReplayBuffer(args.buffer_size, device, energy.log_reward,args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, device, energy.log_reward,args.batch_size, data_ndim=energy.data_ndim, beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    gfn_model.train()
    for i in trange(args.epochs + 1):
        metrics['train/loss'] = train_step(energy, gfn_model, gfn_optimizer, i, args.exploratory,
                                           buffer, buffer_ls, args.exploration_factor, args.exploration_wd)
        
        if i % 1000 == 0:  # todo this logging freq should be an arg
            if False: #not hasattr(energy, "SAMPLE_DISABLED"):
                metrics.update(eval_step(eval_data, energy, gfn_model, final_eval=False))

            if args.energy in ["five_mvn", "four_bananas", "michalewicz", "rastrigin", "rosenbrock", "cancer", "credit"]:
                if jax_eval is None:
                    jax_eval = get_jax_eval(args, energy)

                evalled = jax_eval(gfn_model)
                metrics.update(evalled)

            if args.energy in ["control2d_actions"]:

                with torch.no_grad():
                    gfn_model.eval()
                    samples = gfn_model.sample(10000, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward).cpu()
                    gfn_model.train()

                    rewards = energy.score_batch(samples)
                    metrics["control2d/has_solved"] = (
                                rewards >= 1).any()  # + jax.random.uniform(rng, (1,), minval=0.1, maxval=0.2)
                    metrics["control2d/avg_on_goal"] = torch.vmap(lambda run: (run >= 1).sum())(rewards).float().mean()
                    metrics["control2d/avg_touched_goal"] = torch.vmap(lambda run: (run >= 1).any())(rewards).float().mean()
                    metrics["control2d/avg_stopped_goal"] = torch.vmap(lambda run: (run[-1] >= 1).any())(rewards).float().mean()
                    metrics["control2d/mean_reward"] = rewards.mean()
                    metrics["control2d/std_reward"] = rewards.std()
                    metrics["control2d/max_reward"] = rewards.max()
                    metrics["control2d/min_reward"] = rewards.min()
                    metrics["control2d/reward_sum_mean"] = torch.vmap(lambda run: run.sum())(rewards).float().mean()
                    metrics["control2d/reward_sum_std"] = torch.vmap(lambda run: run.sum())(rewards).float().std()
                    metrics["control2d/reward_sum_max"] = torch.vmap(lambda run: run.sum())(rewards).float().max()
                    metrics["control2d/reward_sum_min"] = torch.vmap(lambda run: run.sum())(rewards).float().min()
                    metrics["control2d/reward_sum_med"] = torch.median(torch.vmap(lambda run: run.sum())(rewards))

                    def nn_diversity(nn_weights):
                        covmat = torch.cov(nn_weights.T)
                        normed = torch.linalg.norm(covmat)
                        return normed

                    metrics["control2d/nn_diversity"] = nn_diversity(samples)



            gfn_model.eval()
            samples = gfn_model.sample(10000, lambda bsz: uniform_discretizer(bsz, args.T), energy.log_reward).cpu()
            min, max = samples.min(), samples.max()
            metrics["metrics/min_sample"] = min
            metrics["metrics/max_sample"] = max
            gfn_model.train()


            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']

            metrics.update(eval_step_K_step_discretizer(energy, gfn_model, final_eval=False))
            # if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
            #     del metrics[f'eval_{args.discretizer_traj_length}_steps/log_Z_learned']

            images = plot_step(energy, gfn_model, name)
            metrics.update(images)
            plt.close('all')
            wandb.log(metrics, step=i)
            if i % 1000 == 0:
                torch.save(gfn_model.state_dict(), f'{name}model.pt')

    path = f"{wandb.run.dir}/gfndiffusion_{args.seed}.csv"
    from tpdist.utils.wandbcsv import save_pd
    save_pd(path)
    artifact = wandb.Artifact(name="csv", type="dataset")
    artifact.add_file(path, name="csv")
    wandb.log_artifact(artifact)

    if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
        del metrics['eval/log_Z_learned']

    if not hasattr(energy, "SAMPLE_DISABLED"):
        eval_results_K = final_eval_K_steps(energy, gfn_model)
        metrics.update(eval_results_K)
    # if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
    #     del metrics[f'eval_{args.discretizer_traj_length}_steps/log_Z_learned']

    #torch.save(gfn_model.state_dict(), f'{name}model_final.pt')


def final_eval(energy, gfn_model):
    final_eval_data = energy.sample(final_eval_data_size).to(device) if not (args.energy in _LIST_OF_NO_SAMPLES_ENERGIES) else None
    results = eval_step(final_eval_data, energy, gfn_model, final_eval=True)
    return results


def final_eval_K_steps(energy, gfn_model):
    final_eval_data = energy.sample(final_eval_data_size).to(device) if not (args.energy in _LIST_OF_NO_SAMPLES_ENERGIES) else None
    results = eval_step_K_step_discretizer(final_eval_data, energy, gfn_model, final_eval=True)
    return results


def eval():
    name = get_name(args)

    print(name)

    energy = get_energy()
    eval_data = energy.sample(eval_data_size).to(device) if not (args.energy in _LIST_OF_NO_SAMPLES_ENERGIES) else None

    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, zero_init=args.zero_init, device=device).to(device)

    model_final_path = name + 'model_final.pt'
    model_path = name + 'model.pt'

    if os.path.exists(model_final_path):
        try:
            gfn_model.load_state_dict(torch.load(model_final_path, weights_only=True))
        except:
            print("Couldn't load final model")
    else:
        if os.path.exists(model_path):
            try:
                gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
            except:
                print("Couldn't load model")
        else:
            print("NO MODEL IS AVAILABLE")
            return

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy - proper evaluation", config=config, name=name)

    print(gfn_model)
    metrics = dict()

    gfn_model.eval()
    for i in trange(1, 201):
        metrics.update(eval_step_K_step_discretizer(eval_data, energy, gfn_model, final_eval=False, traj_length=i))
        # if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
        #     del metrics[f'eval_{i}_steps/log_Z_learned']
    wandb.log(metrics)


if __name__ == '__main__':
    if args.eval:
        eval()
    else:
        train()
