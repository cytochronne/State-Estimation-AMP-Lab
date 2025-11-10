# 用于直观检查最终 obs 的组成与 shape

import torch
from isaaclab.envs import ManagerBasedRLEnv
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotPlayEnvCfg, RobotEnvCfg

def infer_num_joints(env):
    # 多种兼容方式尝试获取关节数
    robot = getattr(env.scene, "robot", None)
    if robot is not None and hasattr(robot, "num_dof"):
        return int(robot.num_dof)
    if hasattr(env.scene, "articulations") and "robot" in env.scene.articulations:
        art = env.scene.articulations["robot"]
        if hasattr(art, "num_dof"):
            return int(art.num_dof)
    # 兜底：从 action 维度倒推
    obs = env.reset()
    act_dim = env.action_manager.action_spec.num_actions
    return int(act_dim)

def infer_height_dim(env):
    # 优先从传感器拿；如不可得，按 size/resolution 估计
    sensor = env.scene.sensors.get("height_scanner", None)
    if sensor is not None:
        # 常见属性尝试
        for attr in ["num_rays", "num_beams", "ray_count"]:
            if hasattr(sensor, attr):
                return int(getattr(sensor, attr))
        # 估算
        pcfg = sensor.cfg.pattern_cfg
        sx, sy = pcfg.size[0], pcfg.size[1]
        res = pcfg.resolution
        nx = int(round(sx / res)) + 1
        ny = int(round(sy / res)) + 1
        return nx * ny
    # 兜底：从拼接向量末尾推断（需知道其它项维度）
    raise RuntimeError("无法从传感器直接推断 n_h，请按下方打印结果人工校验。")

def split_last_frame_terms(vec_last_frame, n_j, n_h, is_critic=False):
    # 按 ObservationsCfg 中定义顺序切片（单帧）
    idx = 0
    out = {}
    def take(k):
        nonlocal idx
        s = vec_last_frame[..., idx:idx+k]
        idx += k
        return s

    out["base_lin_vel"]      = take(3)
    out["base_ang_vel"]      = take(3)
    out["projected_gravity"] = take(3)
    out["velocity_commands"] = take(3)
    out["joint_pos_rel"]     = take(n_j)
    out["joint_vel_rel"]     = take(n_j)
    if is_critic:
        out["joint_effort"]  = take(n_j)
    out["last_action"]       = take(n_j)
    out["height_scanner"]    = take(n_h)
    assert idx == vec_last_frame.shape[-1], f"帧切片未对齐，已取 {idx}, 但总长 {vec_last_frame.shape[-1]}"
    return out

def main():
    cfg = RobotPlayEnvCfg()  # 或 RobotEnvCfg() 用于训练配置
    env = ManagerBasedRLEnv(cfg)

    obs = env.reset()  # obs 是 dict: {"policy": (N, Ao), "critic": (N, Co)}
    policy = obs["policy"]
    critic = obs["critic"]
    N, Ao = policy.shape
    _, Co = critic.shape

    n_j = infer_num_joints(env)
    n_h = infer_height_dim(env)
    H = cfg.observations.policy.history_length

    Ao_per_step = 12 + 3*n_j + n_h
    Co_per_step = 12 + 4*n_j + n_h

    print(f"num_envs={N}, n_j={n_j}, n_h={n_h}, history_length={H}")
    print(f"Ao_per_step={Ao_per_step}, Ao={Ao} (期望 {H * Ao_per_step})")
    print(f"Co_per_step={Co_per_step}, Co={Co} (期望 {H * Co_per_step})")

    # 还原为 (N, H, per_step)
    policy_blocks = policy.view(N, H, Ao_per_step)
    critic_blocks = critic.view(N, H, Co_per_step)
    # 取最新一帧（通常最后一个块）
    pol_last = policy_blocks[:, -1, :]
    cri_last = critic_blocks[:, -1, :]

    # 按单帧顺序切开各项
    pol_terms = split_last_frame_terms(pol_last, n_j, n_h, is_critic=False)
    cri_terms = split_last_frame_terms(cri_last, n_j, n_h, is_critic=True)

    print("\nPolicy(最后一帧)各项shape：")
    for k, v in pol_terms.items():
        print(f"  {k:>18s}: {tuple(v.shape)}")

    print("\nCritic(最后一帧)各项shape：")
    for k, v in cri_terms.items():
        print(f"  {k:>18s}: {tuple(v.shape)}")

    # 验证“最后 n_h 元素”确为最新一帧 height_scanner
    tail_height = policy[:, -n_h:]
    assert torch.allclose(tail_height, pol_terms["height_scanner"], atol=0, rtol=0), \
        "最后 n_h 元素不等于最新一帧的 height_scanner（切分或顺序与假设不符）"
    print("\n校验通过：policy 向量末尾的 n_h 确为最新一帧 height_scanner。")

    # 如需查看每个时间块（历史每一帧）中 height_scanner 的 shape：
    heights_over_time = policy_blocks[..., -n_h:]  # (N, H, n_h)
    print(f"\nheight_scanner over time shape: {tuple(heights_over_time.shape)}  (应为 (N, {H}, n_h))")

if __name__ == "__main__":
    main()