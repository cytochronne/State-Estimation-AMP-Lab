# 特权教师：
## 观测：
下面基于你当前的配置（history_length=5，concatenate_terms=True，包含 height_scanner）给出“实际进入 rsl_rl 的最终 obs”的组成、顺序、每项 shape，以及总维度如何计算，并提供一段可直接运行的检查脚本，打印各成分与最终拼接后的 shape。

一、每步（单帧）原子观测项与维度
记：
- n_j = 机器人关节数（Go2 通常是 12）
- n_h = 高度雷射扫描长度（由 RayCaster 网格决定）
- height_scanner 的网格由 patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) 给出，n_h ≈ (floor(1.6/0.1)+1) * (floor(1.0/0.1)+1) = 17*11 = 187（以实际传感器实现为准，下面给出代码检查）
- 其它项都是标量或固定向量，不改变 shape

按在 ObservationsCfg.PolicyCfg 中定义的顺序（order preserved），单帧各项维度为：
- base_lin_vel: 3
- base_ang_vel: 3
- projected_gravity: 3
- velocity_commands: 3
- joint_pos_rel: n_j
- joint_vel_rel: n_j
- last_action: n_j
- height_scanner: n_h

因此，单帧“policy”观测维度
- Ao_per_step = 3+3+3+3 + n_j + n_j + n_j + n_h = 12 + 3*n_j + n_h
- 对 Go2（n_j=12）：Ao_per_step = 48 + n_h

“critic”组（比 policy 多了 joint_effort）单帧维度
- Co_per_step = 3+3+3+3 + n_j + n_j + n_j + n_j + n_h = 12 + 4*n_j + n_h
- 对 Go2：Co_per_step = 60 + n_h

二、history_length=5 与拼接后的最终 obs
- 由于 ObservationsCfg.PolicyCfg.history_length = 5 且 concatenate_terms = True，管理器会把最近 5 帧按时间顺序串接在最后一个维度上（通常从最旧到最新，最后一段为当前帧）。
- 最终 policy/critic 的观测维度：
  - Ao = history_length * Ao_per_step = 5 * (12 + 3*n_j + n_h)
  - Co = history_length * Co_per_step = 5 * (12 + 4*n_j + n_h)

三、与 TerrainAwareRecurrentActorCritic 的对接与切分
- 该网络把“最后 height_obs_dim 个元素”当作地形高度输入 height_scanner，并走地形 MLP 分支；其余作为“core”走 RNN 和直连。
- 因为 history 拼接在最后维度上，且 height_scanner 在每帧的最后一项，故“最后 n_h 个元素”恰好对应“最新一帧的 height_scanner”（这正是模型期望的高度输入）。
- 注意：前 4 帧的 height_scanner 会落在“core”那一大段里，一并输入 RNN（这是该实现的设计选择：仅最新帧走地形 MLP，历史的地形数据进入 RNN）。

因此，给 TerrainAwareRecurrentActorCritic 的期望输入形式与 shape 为：
- act(observations): observations 形状 (B, Ao) 或 (T, B, Ao)
- evaluate(critic_observations): 形状 (B, Co) 或 (T, B, Co)
- 其中 Ao 与 Co 按上一节公式计算；height_obs_dim 应设置为 n_h（不是 5*n_h）

四、如何直观检查各成分 shape 与最终 obs shape
下面的脚本会：
- 创建环境并 reset
- 推断 n_j 与 n_h
- 计算 Ao_per_step/Co_per_step 与 Ao/Co
- 把 policy/critic 的拼接观测按“时间块”（history_length 个 block）与“项内顺序”切片，打印每项在“最新一帧”的切片 shape，并验证“最后 n_h 元素”与“最新一帧的 height_scanner 切片”一致

你可以将其保存为一个临时脚本并在项目根目录运行（或粘到一个 Notebook 单元运行）。

````python
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
````

五、要点小结
- 最终 obs 顺序是“按帧拼接、帧内按 term 顺序拼接”，height_scanner 在每帧末尾；因此 obs 向量的“最后 n_h 个元素”对应“最新一帧的 height_scanner”，可直接作为 height_obs_dim 输入给 TerrainAwareRecurrentActorCritic。
- 前 H-1 帧的 height_scanner 会被包含在“core”部分，进入 RNN（设计上用 RNN建模历史地形+本体信息；仅最新地形经过专用 MLP）。
- Go2 常见设置下：Ao_per_step = 48 + n_h，Co_per_step = 60 + n_h；最终 Ao=5*(48+n_h)、Co=5*(60+n_h)。实际 n_h 请以脚本打印为准。
