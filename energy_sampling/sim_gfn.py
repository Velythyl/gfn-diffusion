import os
import pathlib
import random
import signal
import sys

import hydra
import wandb
from omegaconf import OmegaConf


def run_gfndiffusion_instead(cfg):

    distname = cfg.dist.distfam
    if distname == "multi_mvn":
        distname = "five_mvn"
    elif distname == "multi_dist":
        distname = "four_bananas"

    if cfg.wandb.seed == -1:
        cfg.wandb.seed = random.randint(0, 20000)

    sys.argv = ["train.py"]  # First argument is the script name
    sys.argv += [f"--energy={distname}", f"--energy_dim={cfg.dist.dim}", f"--seed={cfg.wandb.seed}", f"--wandb_name={cfg.wandb.project}", f"--epochs={cfg.adapt.epochs}"]  # Add key-value pairs as arguments

    sys.argv += ['--t_scale=1.',  '--pis_architectures', '--zero_init', '--clipping', '--mode_fwd=tb', '--lr_policy=1e-3', '--lr_flow=1e-1', '--exploratory', '--exploration_wd', '--exploration_factor=0.2', '--langevin']

    #sys.argv += ['--t_scale=1.', '--pis_architectures', '--zero_init', '--clipping',
    #  '--mode_fwd=tb', '--lr_policy=1e-3', '--lr_back=1e-3', '--lr_flow=1e-1',
    # '--exploratory', '--exploration_wd', '--exploration_factor=0.1', '--both_ways',
    # '--buffer_size=600000', '--prioritized=rank', '--rank_weight=0.01', '--ld_step=0.1', '--ld_schedule',
    # '--target_acceptance_rate=0.574']

    from train import train
    train()
    # Call the secondary.py logic
    #secondary_main()


@hydra.main(version_base=None, config_path="config", config_name="conf")
def main(cfg):
    assert cfg.adapt.adapt_strat == "GFN Diffusion"

    def kill_em(self, signum: int, frame = None) -> None:
        os.kill(os.getpid(),signal.SIGKILL) # elevate to sigkill
    signal.signal(signal.SIGTERM, kill_em)
    signal.signal(signal.SIGUSR2, kill_em)

    run_gfndiffusion_instead(cfg)
    with open(f"{wandb.run.dir}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    exit()

if __name__ == '__main__':
    main()
