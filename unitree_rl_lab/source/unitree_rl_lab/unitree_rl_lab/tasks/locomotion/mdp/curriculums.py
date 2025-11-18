from __future__ import annotations

import torch
import os
import json
import datetime
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 地形类型定义
TERRAIN_TYPES = [
    "random_rough",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "boxes",
    "pyramid_stairs",
    "pyramid_stairs_inv"
]


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def enhanced_terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    success_threshold: float = 1.0,
    increase_threshold: float = 0.9,
    decrease_threshold: float = 0.5,
    increase_steps: int = 2000,
    decrease_steps: int = 500,
    terrain_type_transition_prob: float = 0.3,
    enable_terrain_transition_tracking: bool = True,
    data_save_dir: str = "./terrain_transition_data",
    terrain_transition_buffer_size: int = 1000
) -> torch.Tensor:
    """
    增强版的地形级别课程学习函数，支持地形类型切换和地形切换数据收集。
    
    Args:
        env: 环境实例
        env_ids: 需要更新的环境ID
        success_threshold: 成功阈值
        increase_threshold: 增加难度的阈值
        decrease_threshold: 降低难度的阈值
        increase_steps: 增加难度所需步数
        decrease_steps: 降低难度所需步数
        terrain_type_transition_prob: 地形类型切换概率
        enable_terrain_transition_tracking: 是否启用地形切换跟踪
        data_save_dir: 数据保存目录
        terrain_transition_buffer_size: 缓冲区大小
    
    Returns:
        更新后的地形级别
    """
    # 确保环境具有必要的地形跟踪变量
    if not hasattr(env, "current_terrain_levels"):
        env.current_terrain_levels = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "previous_terrain_levels"):
        env.previous_terrain_levels = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "terrain_switched"):
        env.terrain_switched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "terrain_types"):
        env.terrain_types = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    
    # 初始化地形级别（如果不存在）
    if not hasattr(env, "terrain_levels"):
        env.terrain_levels = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    
    # 保存先前的地形级别
    env.previous_terrain_levels[:] = env.terrain_levels.clone()
    
    # 调用原始的地形级别课程学习逻辑（基于isaaclab的实现思想）
    # 这里实现一个基于成功率的课程学习逻辑
    if hasattr(env.reward_manager, "_episode_sums"):
        # 假设我们使用track_lin_vel_xy奖励来衡量成功
        reward_term_name = "track_lin_vel_xy"
        if reward_term_name in env.reward_manager._episode_sums:
            # 计算平均奖励作为成功指标
            avg_reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
            
            # 获取奖励项配置
            reward_term = env.reward_manager.get_term_cfg(reward_term_name)
            success_rate = avg_reward / (reward_term.weight * success_threshold)
            
            # 根据成功率调整地形级别
            for env_id in env_ids:
                if env.common_step_counter % increase_steps == 0 and success_rate[env_id] > increase_threshold:
                    # 增加地形难度
                    env.terrain_levels[env_id] = min(env.terrain_levels[env_id] + 1, 10)
                elif env.common_step_counter % decrease_steps == 0 and success_rate[env_id] < decrease_threshold:
                    # 降低地形难度
                    env.terrain_levels[env_id] = max(env.terrain_levels[env_id] - 1, 0)
    
    # 更新当前地形级别
    env.current_terrain_levels[:] = env.terrain_levels.clone()
    
    # 检测地形级别变化
    env.terrain_switched[:] = env.current_terrain_levels != env.previous_terrain_levels
    
    # 地形类型切换逻辑
    if terrain_type_transition_prob > 0:
        # 生成随机数，决定是否切换地形类型
        transition_mask = torch.rand(env.num_envs, device=env.device) < terrain_type_transition_prob
        
        # 为需要切换的环境生成新的地形类型
        if torch.any(transition_mask):
            # 为每个需要切换的环境随机选择一种地形类型
            new_terrain_types = torch.randint(0, len(TERRAIN_TYPES), (torch.sum(transition_mask).item(),), device=env.device)
            env.terrain_types[transition_mask] = new_terrain_types
            # 同时标记这些环境为地形切换
            env.terrain_switched[transition_mask] = True
    
    # 如果启用地形切换跟踪，初始化数据收集相关变量
    if enable_terrain_transition_tracking:
        if not hasattr(env, "terrain_transition_buffer"):
            env.terrain_transition_buffer = []
        if not hasattr(env, "data_save_dir"):
            env.data_save_dir = data_save_dir
            os.makedirs(env.data_save_dir, exist_ok=True)
        if not hasattr(env, "buffer_size"):
            env.buffer_size = terrain_transition_buffer_size
        
        # 收集地形切换数据
        switch_env_ids = torch.where(env.terrain_switched)[0]
        if len(switch_env_ids) > 0:
            _collect_terrain_transition_data(env, switch_env_ids)
    
    return env.terrain_levels


def _collect_terrain_transition_data(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    收集地形切换时的状态数据。
    
    Args:
        env: 环境实例
        env_ids: 发生地形切换的环境ID
    """
    try:
        # 获取当前观察
        current_obs = env._get_observations()["policy"].detach().clone()
        
        # 收集切换数据
        for env_id in env_ids:
            # 尝试获取机器人状态
            robot_state = None
            if hasattr(env, "robot") and hasattr(env.robot, "data"):
                if hasattr(env.robot.data, "root_pos_w"):
                    # 收集基本的机器人状态
                    root_pos = env.robot.data.root_pos_w[env_id].cpu().numpy()
                    root_quat = env.robot.data.root_quat_w[env_id].cpu().numpy()
                    root_vel = env.robot.data.root_lin_vel_w[env_id].cpu().numpy()
                    robot_state = {
                        "root_pos": root_pos.tolist(), 
                        "root_quat": root_quat.tolist(), 
                        "root_vel": root_vel.tolist()
                    }
            
            transition_data = {
                "env_id": env_id.item(),
                "previous_terrain_level": env.previous_terrain_levels[env_id].item(),
                "current_terrain_level": env.current_terrain_levels[env_id].item(),
                "terrain_type": env.terrain_types[env_id].item(),
                "terrain_type_name": TERRAIN_TYPES[env.terrain_types[env_id].item()] if env.terrain_types[env_id].item() < len(TERRAIN_TYPES) else "unknown",
                "observation": current_obs[env_id].cpu().numpy().tolist(),
                "robot_state": robot_state,
                "timestamp": env.sim.current_time if hasattr(env, "sim") else 0,
                "step": env.common_step_counter
            }
            
            # 添加到缓冲区
            env.terrain_transition_buffer.append(transition_data)
            
            # 如果缓冲区过大，保存到文件并清空
            if len(env.terrain_transition_buffer) >= env.buffer_size:
                _save_terrain_transition_data(env)
    except Exception as e:
        print(f"Error collecting terrain transition data: {e}")


def _save_terrain_transition_data(env: ManagerBasedRLEnv):
    """
    将地形切换数据保存到文件。
    
    Args:
        env: 环境实例
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"terrain_transitions_{timestamp}.json"
        filepath = os.path.join(env.data_save_dir, filename)
        
        # 转换为可JSON序列化的格式
        serializable_data = []
        for data in env.terrain_transition_buffer:
            serializable_data.append({
                "env_id": data["env_id"],
                "previous_terrain_level": data["previous_terrain_level"],
                "current_terrain_level": data["current_terrain_level"],
                "terrain_type": data["terrain_type"],
                "terrain_type_name": data["terrain_type_name"],
                "observation": data["observation"],
                "robot_state": data["robot_state"],
                "timestamp": float(data["timestamp"]),
                "step": data["step"]
            })
        
        # 保存到文件
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f)
        
        print(f"Saved {len(serializable_data)} terrain transition samples to {filepath}")
        
        # 清空缓冲区
        env.terrain_transition_buffer = []
    except Exception as e:
        print(f"Error saving terrain transition data: {e}")
