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
parser.add_argument("--debug_file", type=str, required=True, help="Path to the debug data .pt file.")

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

from rsl_rl_AEMP.runners import OnPolicyRunner

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
    
    # =================================================================================
    # LOAD DEBUG DATA
    # =================================================================================
    print(f"[INFO] Loading debug data from: {args_cli.debug_file}")
    try:
        debug_data = torch.load(args_cli.debug_file, map_location=agent_cfg.device)
        saved_obs = debug_data["obs"].to(agent_cfg.device)
        saved_priv_obs = debug_data["privileged_obs"].to(agent_cfg.device)
        
        # Load latents and scores for comparison
        saved_student_latent = debug_data.get("student_latent")
        if saved_student_latent is not None: saved_student_latent = saved_student_latent.to(agent_cfg.device)
        
        saved_teacher_latent = debug_data.get("teacher_latent")
        if saved_teacher_latent is not None: saved_teacher_latent = saved_teacher_latent.to(agent_cfg.device)
        
        saved_student_score = debug_data.get("student_score")
        if saved_student_score is not None: saved_student_score = saved_student_score.to(agent_cfg.device)
        
        saved_teacher_score = debug_data.get("teacher_score")
        if saved_teacher_score is not None: saved_teacher_score = saved_teacher_score.to(agent_cfg.device)

        print(f"[INFO] Loaded debug data. Obs shape: {saved_obs.shape}, PrivObs shape: {saved_priv_obs.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load debug data: {e}")
        return
    # =================================================================================

    # Ensure policy is in eval mode
    runner.eval_mode()

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            
            # =================================================================================
            # OVERRIDE OBSERVATIONS WITH DEBUG DATA
            # =================================================================================
            # Ensure we match the environment's num_envs
            current_num_envs = env.num_envs
            
            # Slice or repeat saved data to match current_num_envs
            if saved_obs.shape[0] >= current_num_envs:
                obs_override = saved_obs[:current_num_envs]
                priv_obs_override = saved_priv_obs[:current_num_envs]
                
                # Prepare comparison data
                curr_saved_student_latent = saved_student_latent[:current_num_envs] if saved_student_latent is not None else None
                curr_saved_teacher_latent = saved_teacher_latent[:current_num_envs] if saved_teacher_latent is not None else None
                curr_saved_student_score = saved_student_score[:current_num_envs] if saved_student_score is not None else None
                curr_saved_teacher_score = saved_teacher_score[:current_num_envs] if saved_teacher_score is not None else None
            else:
                # Repeat if not enough
                repeat_factor = (current_num_envs // saved_obs.shape[0]) + 1
                obs_override = saved_obs.repeat(repeat_factor, 1)[:current_num_envs]
                priv_obs_override = saved_priv_obs.repeat(repeat_factor, 1)[:current_num_envs]
                
                # Prepare comparison data
                curr_saved_student_latent = saved_student_latent.repeat(repeat_factor, 1)[:current_num_envs] if saved_student_latent is not None else None
                curr_saved_teacher_latent = saved_teacher_latent.repeat(repeat_factor, 1)[:current_num_envs] if saved_teacher_latent is not None else None
                curr_saved_student_score = saved_student_score.repeat(repeat_factor, 1)[:current_num_envs] if saved_student_score is not None else None
                curr_saved_teacher_score = saved_teacher_score.repeat(repeat_factor, 1)[:current_num_envs] if saved_teacher_score is not None else None
            
            # Apply override
            obs = obs_override
            if "observations" not in extras:
                extras["observations"] = {}
            extras["observations"]["teacher"] = priv_obs_override
            # =================================================================================

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
                # NOTE: We ignore the returned obs here because we will override it in the next loop iteration
                # But for the current loop's printing logic below, we should use the OVERRIDDEN obs, not the new one from env.step()
                # However, play_distill updates `obs` here for the NEXT loop.
                # Since we override at the start of the loop, it doesn't matter what we get here.
                # BUT, for the printing logic below (discriminator score), we want to use the obs we just used for action.
                # So we should NOT update `obs` variable with the result of env.step() if we want to print scores for the current frame.
                # Actually, play_distill prints scores for the NEW obs (after step).
                # If we want to debug the saved data, we should print scores for the SAVED data.
                # So let's capture the new obs in a temp variable, but keep `obs` as the overridden one for printing.
                _new_obs, _, term, _new_extras = ret
            else:
                _new_obs, _, term, _new_extras = ret
            
            # Update extras for metrics (like velocity tracking) from the real env step
            # But keep observations overridden for the printing logic
            # Actually, if we want to print scores for the saved data, we should use `obs` (which is overridden).
            
            # print("INFO:obs", obs)
            # teacher_obs_curr = extras["observations"]["teacher"]
            # print("INFO: priv_obs", teacher_obs_curr)

            # Student encodes and discriminator scores (print AFTER step so state is current)
            if discriminator is not None:
                print("\n" + "="*30 + " DEBUG DATA COMPARISON " + "="*30)
                try:
                    # Use the overridden obs
                    student_latent = policy.get_student_latent(obs)
                    score = discriminator.classify(student_latent)
                    disc = score.mean().item()
                    
                    # Compare Student Latent
                    if curr_saved_student_latent is not None:
                        diff = (student_latent - curr_saved_student_latent).abs().mean().item()
                        print(f"[Student Latent] Saved Mean: {curr_saved_student_latent.mean():.4f}, Std: {curr_saved_student_latent.std():.4f}")
                        print(f"[Student Latent] Curr  Mean: {student_latent.mean():.4f}, Std: {student_latent.std():.4f}")
                        print(f"[Student Latent] Diff (L1): {diff:.6f} {'✅' if diff < 1e-5 else '❌'}")
                    
                    # Compare Student Score
                    if curr_saved_student_score is not None:
                        diff_score = (score - curr_saved_student_score).abs().mean().item()
                        print(f"[Student Score ] Saved Mean: {curr_saved_student_score.mean():.4f}")
                        print(f"[Student Score ] Curr  Mean: {score.mean():.4f}")
                        print(f"[Student Score ] Diff (L1): {diff_score:.6f} {'✅' if diff_score < 1e-5 else '❌'}")

                except Exception as e:
                    print(f"[ERROR] Student comparison failed: {e}")
                    disc = float('nan')

                # Teacher encodes and discriminator scores
                try:
                    if "observations" in extras and "teacher" in extras["observations"]:
                        teacher_obs_curr = extras["observations"]["teacher"]
                        if hasattr(policy, "evaluate_feature"):
                            teacher_latent = policy.evaluate_feature(teacher_obs_curr)
                            teacher_score = discriminator.classify(teacher_latent)
                            disc_teacher = teacher_score.mean().item()
                            
                            # Compare Teacher Latent
                            if curr_saved_teacher_latent is not None:
                                diff = (teacher_latent - curr_saved_teacher_latent).abs().mean().item()
                                print(f"[Teacher Latent] Saved Mean: {curr_saved_teacher_latent.mean():.4f}, Std: {curr_saved_teacher_latent.std():.4f}")
                                print(f"[Teacher Latent] Curr  Mean: {teacher_latent.mean():.4f}, Std: {teacher_latent.std():.4f}")
                                print(f"[Teacher Latent] Diff (L1): {diff:.6f} {'✅' if diff < 1e-5 else '❌'}")

                            # Compare Teacher Score
                            if curr_saved_teacher_score is not None:
                                diff_score = (teacher_score - curr_saved_teacher_score).abs().mean().item()
                                print(f"[Teacher Score ] Saved Mean: {curr_saved_teacher_score.mean():.4f}")
                                print(f"[Teacher Score ] Curr  Mean: {teacher_score.mean():.4f}")
                                print(f"[Teacher Score ] Diff (L1): {diff_score:.6f} {'✅' if diff_score < 1e-5 else '❌'}")

                        else:
                            disc_teacher = float('nan')
                    else:
                        disc_teacher = float('nan')
                except Exception as e:
                    print(f"[ERROR] Teacher comparison failed: {e}")
                    disc_teacher = float('nan')
                
                print("="*83)

                # Compute velocity tracking metrics from env state
                # We use _new_extras here because these contain the info from the physics step we just performed
                acc_lin = _new_extras.get("VelTracking/accuracy_lin_xy", None) if isinstance(_new_extras, dict) else None
                succ = _new_extras.get("VelTracking/success_rate", None) if isinstance(_new_extras, dict) else None
                
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

                parts = [f"Discriminator: {disc:.4f}"]
                if not torch.isnan(torch.tensor(disc_teacher)):
                    parts.append(f"DiscTeacher: {disc_teacher:.4f}")
                if acc_lin is not None:
                    parts.append(f"AccLin: {float(acc_lin):.3f}")
                if succ is not None:
                    parts.append(f"Success: {float(succ):.3f}")
                print(" | ".join(parts), end="\r")
            
            # Update extras for next loop (though we will override observations again)
            extras = _new_extras if isinstance(_new_extras, dict) else {}

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
