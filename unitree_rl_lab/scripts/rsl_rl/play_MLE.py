# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl_MLE.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join(args_cli.log_root, "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    # Always use OnPolicyRunner for AEMP as it handles Distillation
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    print(f"[CHECK] Checkpoint path: {resume_path}")


    # obtain the trained policy for inference
    # Access policy and discriminator directly from the algorithm
    policy = runner.alg.policy
    discriminator = getattr(runner.alg, "discriminator", None)
    
    if discriminator is None:
        print("[WARN] No discriminator found in the loaded model. Visualization will be skipped.")

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    #export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    ret = env.get_observations()
    if isinstance(ret, tuple):
        obs, extras = ret
    else:
        obs = ret
        extras = {};

    timestep = 0
    # accumulate success over episode for normalized success rate
    succ_sum = 0.0
    # parameters from reward configs if available
    try:
        rm = getattr(env.unwrapped, "reward_manager", None)
        track_cfg = rm.get_term_cfg("track_lin_vel_xy") if rm is not None else None
        std_lin = float(track_cfg.params.get("std", 0.5)) if track_cfg is not None else 0.5
        succ_cfg = rm.get_term_cfg("vel_tracking_success") if rm is not None else None
        lin_thresh = float(succ_cfg.params.get("lin_thresh", 0.1)) if succ_cfg is not None else 0.1
        max_len = getattr(env.unwrapped, "max_episode_length", None)
        if isinstance(max_len, torch.Tensor):
            max_len = float(max_len.mean().item())
        elif max_len is not None:
            max_len = float(max_len)
    except Exception:
        std_lin, lin_thresh, max_len = 0.5, 0.1, None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Teacher acts
            if "observations" in extras and "teacher" in extras["observations"]:
                teacher_obs = extras["observations"]["teacher"]
                actions = policy.evaluate(teacher_obs)
            else:
                # Fallback if teacher obs not found (should not happen if configured correctly)
                actions = policy.act(obs)

            # env stepping
            ret = env.step(actions)
            if isinstance(ret, tuple) and len(ret) == 4:
                obs, _, term, extras = ret
            else:
                obs, _, term, extras = ret

            # print("INFO:obs", obs)
            # teacher_obs_curr = extras["observations"]["teacher"]
            # print("INFO: priv_obs", teacher_obs_curr)

            # Student encodes and discriminator scores (print AFTER step so state is current)
            student_mean = None
            uncertainty_val = None
            try:
                ret = policy.get_student_latent(obs)
                if isinstance(ret, tuple):
                    student_mean, student_cov = ret
                    # cov is sigma^2, we display mean std dev as uncertainty
                    uncertainty_val = student_cov.sqrt().mean().item()
                else:
                    student_mean = ret
            except Exception:
                pass

            disc = float('nan')
            disc_teacher = float('nan')

            if discriminator is not None:
                try:
                    if student_mean is not None:
                        score = discriminator.classify(student_mean)
                        disc = score.mean().item()
                except Exception:
                    pass

                # Teacher encodes and discriminator scores
                try:
                    if "observations" in extras and "teacher" in extras["observations"]:
                        teacher_obs_curr = extras["observations"]["teacher"]
                        if hasattr(policy, "evaluate_feature"):
                            #print("=====================teacher_latent=====================")
                            teacher_latent = policy.evaluate_feature(teacher_obs_curr)
                            teacher_score = discriminator.classify(teacher_latent)
                            disc_teacher = teacher_score.mean().item()
                except Exception:
                    pass

            # Compute velocity tracking metrics from env state
            acc_lin = extras.get("VelTracking/accuracy_lin_xy", None)
            succ = extras.get("VelTracking/success_rate", None)
            try:
                asset = env.unwrapped.scene["robot"]
                cmd_b = env.unwrapped.command_manager.get_command("base_velocity")
                vel_b = asset.data.root_lin_vel_b[:, :2]
                err_lin = torch.linalg.norm(vel_b - cmd_b[:, :2], dim=1)
                acc_lin_inst = torch.exp(-err_lin / (std_lin**2)).mean().item()
                acc_lin = acc_lin if acc_lin is not None else max(0.0, min(1.0, float(acc_lin_inst)))
                succ_step = (err_lin < lin_thresh).float().mean().item()
                if max_len:
                    succ_sum += float(succ_step)
                    # reset on episode termination
                    try:
                        if hasattr(term, "any") and term.any():
                            succ_sum = 0.0
                    except Exception:
                        pass
                    succ_norm = max(0.0, min(1.0, succ_sum / max_len))
                    succ = succ if succ is not None else succ_norm
            except Exception:
                pass

            parts = []
            if discriminator is not None:
                parts.append(f"Discriminator: {disc:.4f}")
                if not torch.isnan(torch.tensor(disc_teacher)):
                    parts.append(f"DiscTeacher: {disc_teacher:.4f}")
            
            if uncertainty_val is not None:
                parts.append(f"Uncertainty: {uncertainty_val:.4f}")

            if acc_lin is not None:
                parts.append(f"AccLin: {float(acc_lin):.3f}")
            if succ is not None:
                parts.append(f"Success: {float(succ):.3f}")
            print(" | ".join(parts), end="\r")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
