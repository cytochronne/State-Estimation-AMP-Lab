# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    arg_group.add_argument(
        "--log_root", type=str, default="logs", help="Root folder for logs"
    )

    # uncertainty logging arguments (aligned with rsl_rl group)
    arg_group.add_argument(
        "--uncertainty_method",
        type=str,
        default="mc_dropout",
        choices={"mc_dropout", "ensemble", "mc_ensemble", "none"},
        help="Uncertainty estimation method for logging",
    )
    arg_group.add_argument("--uncertainty_num_passes", type=int, default=10, help="MC passes per model")
    arg_group.add_argument("--uncertainty_num_models", type=int, default=1, help="Number of ensemble models")
    arg_group.add_argument("--uncertainty_dropout", type=float, default=0.1, help="Dropout probability")
    arg_group.add_argument(
        "--uncertainty_weight_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to ensemble weights",
    )
    arg_group.add_argument("--uncertainty_sample_size", type=int, default=256, help="Sample size for logging")


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if rslrl_cfg.experiment_name == "":
        rslrl_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    if agent_cfg.experiment_name == "":
        task_name = args_cli.task
        agent_cfg.experiment_name = task_name.lower().replace("-", "_").removesuffix("_play")

    # propagate uncertainty config
    uc = getattr(agent_cfg, "uncertainty_cfg", None) or {}
    if hasattr(args_cli, "uncertainty_method") and args_cli.uncertainty_method is not None:
        uc["method"] = args_cli.uncertainty_method
    if hasattr(args_cli, "uncertainty_num_passes") and args_cli.uncertainty_num_passes is not None:
        uc["num_passes"] = args_cli.uncertainty_num_passes
    if hasattr(args_cli, "uncertainty_num_models") and args_cli.uncertainty_num_models is not None:
        uc["num_models"] = args_cli.uncertainty_num_models
    if hasattr(args_cli, "uncertainty_dropout") and args_cli.uncertainty_dropout is not None:
        uc["dropout_prob"] = args_cli.uncertainty_dropout
    if hasattr(args_cli, "uncertainty_weight_noise_std") and args_cli.uncertainty_weight_noise_std is not None:
        uc["weight_noise_std"] = args_cli.uncertainty_weight_noise_std
    if hasattr(args_cli, "uncertainty_sample_size") and args_cli.uncertainty_sample_size is not None:
        uc["sample_size"] = args_cli.uncertainty_sample_size
    setattr(agent_cfg, "uncertainty_cfg", uc)

    return agent_cfg
