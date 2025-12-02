import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

import cli_args  

parser = argparse.ArgumentParser(description="Play RSL-RL agent with MC Dropout uncertainty.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--mc_passes", type=int, default=10, help="Number of MC Dropout forward passes per frame.")
parser.add_argument("--mc_dropout_prob", type=float, default=0.1, help="Dropout probability used for MC sampling.")
parser.add_argument("--ci_alpha", type=float, default=0.95, help="Confidence level for CI output.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import math
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _prepare_actor_input(policy_module, obs_tensor):
    if hasattr(policy_module, "_prepare_features") and hasattr(policy_module, "actor_fusion_encoder"):
        return policy_module._prepare_features(
            obs_tensor.detach().clone(), getattr(policy_module, "actor_height_dim", 0), policy_module.actor_fusion_encoder
        )
    return obs_tensor.detach().clone()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy_infer = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    ret = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, extras = ret
    else:
        obs = ret
        extras = {}

    mc_passes = int(getattr(args_cli, "mc_passes", 10) or 10)

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy_infer(obs)
            ret = env.step(actions)
            if version("rsl-rl-lib").startswith("2.3."):
                obs, _, _, extras = ret
            else:
                obs, _, _, extras = ret

            runner.alg.policy.actor.train()
            input_tensor = _prepare_actor_input(runner.alg.policy, obs)
            um = getattr(runner, "_uncertainty_monitor", None)
            if um is not None:
                um_metrics = um.predict_uncertainty(input_tensor.detach(), num_passes=mc_passes)
                um_total = float(torch.mean(um_metrics["total_var"]).item())
                um_model = float(torch.mean(um_metrics["model_var"]).item())
                um_data = float(torch.mean(um_metrics["data_var"]).item())
                msg = f"ReconVar(mean): {um_total:.4f} (model {um_model:.4f}, data {um_data:.4f}) | Samples: {mc_passes}"
            else:
                msg = f"ReconVar(mean): N/A | Samples: {mc_passes}"
            print(msg, end="\r")
            runner.alg.policy.actor.eval()

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
