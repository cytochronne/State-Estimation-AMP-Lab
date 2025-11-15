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

三、与 TerrainAwareActorCritic 的对接与切分
- 该网络把“最后 height_obs_dim 个元素”当作地形高度输入 height_scanner，并走地形 MLP 分支；其余作为“core”走 RNN 和直连。
- 因为 history 拼接在最后维度上，且 height_scanner 在每帧的最后一项，故“最后 n_h 个元素”恰好对应“最新一帧的 height_scanner”（这正是模型期望的高度输入）。
- 注意：前 4 帧的 height_scanner 会落在“core”那一大段里，一并输入 RNN（这是该实现的设计选择：仅最新帧走地形 MLP，历史的地形数据进入 RNN）。

因此，给 TerrainAwareActorCritic 的期望输入形式与 shape 为：
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
- 最终 obs 顺序是“按帧拼接、帧内按 term 顺序拼接”，height_scanner 在每帧末尾；因此 obs 向量的“最后 n_h 个元素”对应“最新一帧的 height_scanner”，可直接作为 height_obs_dim 输入给 TerrainAwareActorCritic。
- 前 H-1 帧的 height_scanner 会被包含在“core”部分，进入 RNN（设计上用 RNN建模历史地形+本体信息；仅最新地形经过专用 MLP）。
- Go2 常见设置下：Ao_per_step = 48 + n_h，Co_per_step = 60 + n_h；最终 Ao=5*(48+n_h)、Co=5*(60+n_h)。实际 n_h 请以脚本打印为准。


# pad与RNN
好，我用 **最好理解的例子（含图解）** 帮你理解 **pad 的作用**，特别是在 IsaacLab / RSL-RL / RNN Memory 中处理 **历史序列时**为什么需要 pad & unpad。

---

## 🌟 问题背景：不同环境在不同时间 done，历史长度不一致

假设你有 3 个并行环境（env0 / env1 / env2），你想给 RNN 输入最近 **3 步观测历史**：

```
history_length = 3   # T = 3
```

真实情况可能是：

| time step | env0 | env1   | env2                 |
| --------- | ---- | ------ | -------------------- |
| t-2       | 有观测  | 有观测    | ✅ done (episode 重启了) |
| t-1       | 有观测  | ✅ done | ✅ done               |
| t         | 有观测  | 有观测    | 有观测                  |

> 也就是说，**某些环境的历史不够长**（因为中间 done），导致历史数据缺失。

但 **RNN 要求输入 shape 必须严格一致：**

```
(T, num_envs, obs_dim)
```

即使 env1、env2 的历史缺失，也必须送进去一个同 shape 的 tensor。

---

## ✅ pad 就是用来“补历史”的

对缺失的时间步，用某个值（通常是 0）填满：

```
原始历史 (未 pad)：
env0: [o(t-2), o(t-1), o(t)]
env1: [o(t-2), ——, o(t)]
env2: [——, ——, o(t)]
```

pad 后得到统一 shape (3, 3, D) 的 tensor：

```
       t-2        t-1        t
-------------------------------------------------
env0 | o0(t-2)    o0(t-1)    o0(t)
env1 | o1(t-2)    PAD        o1(t)
env2 | PAD        PAD        o2(t)
```

用示意图：

```
Before pad (ragged):
[
  env0: [A, B, C]
  env1: [D,   , F]
  env2: [  ,   , G]
]

After pad:
[
  [A, D, PAD],   # t-2
  [B, PAD, PAD], # t-1
  [C, F, G],     # t
]
shape => (T=3, num_envs=3, obs_dim)
```

---

## ✅ mask 表示哪些是 pad，哪些是有效

同时生成一个 mask：

```
mask: 1 表示有效，0 表示 pad

mask = [
  [1, 1, 0],    # t-2
  [1, 0, 0],    # t-1
  [1, 1, 1],    # t
]
```

---

## 🎯 RNN 如何用 pad + mask？

RNN 会：

✔ 对 mask = 0 的 time-step **不更新 hidden state**（防止坏信息污染 RNN）
✔ 对 mask = 1 的 time-step 正常更新

因此 env1 和 env2 的历史缺失**不会让 RNN 误以为它们有连续历史**。

---

## ✅ unpad 的作用（还原回单步输出）

在你的模型里看到：

```python
height_feat = unpad_trajectories(encoded, masks)
```

意思是：

> 训练 RNN 用 pad 的序列，
> 但最后我们只需要 **每个 env 的最后有效时间步** 的 feature。

使用 mask 去掉 pad 后输出：

```
unpad 结果：
env0 -> C
env1 -> F
env2 -> G
```

shape：

```
(B, feature_dim)
```

---

## 🚀 总结（一句话）

| 操作        | 作用                                                          |
| --------- | ----------------------------------------------------------- |
| **pad**   | 让 ragged history 变为统一 tensor `(T, num_envs, D)`，方便 RNN 批量处理 |
| **mask**  | 标记哪些 timestep 是真实的、哪些是 pad（避免污染 RNN）                        |
| **unpad** | 从 RNN 输出中只保留 **最近有效的一步**（一般作为 feature）                      |

---

## 📌 小实验（可直接运行验证）

```python
import torch

obs_dim = 4
T = 3
B = 3

padded = torch.tensor([
    [[1,1,1,1], [2,2,2,2], [0,0,0,0]], # t-2
    [[3,3,3,3], [0,0,0,0], [0,0,0,0]], # t-1
    [[4,4,4,4], [5,5,5,5], [6,6,6,6]], # t
])

masks = torch.tensor([
    [1,1,0],
    [1,0,0],
    [1,1,1]
])

# 保留最后有效
result = []
for env in range(B):
    valid = padded[:, env][masks[:,env]==1]
    result.append(valid[-1])

result = torch.stack(result)
print(result)
```

输出：

```
tensor([[4,4,4,4],
        [5,5,5,5],
        [6,6,6,6]])
```

这就是 unpad 的作用。

---

prepare_features里的observation是经过padding的吗

# 观察与extras
先给你一个结论版，然后再一点点拆开👇

---

## 1. `obs, extras = self.env.get_observations()` 里的 `extras` 到底是啥？

在 **IsaacLab + RSL-RL** 的封装里：

* `obs`：给 **“主策略”/学生 (policy)** 用的观测（通常就是 `ObservationsCfg.PolicyCfg` 这组，被拼成一个 `(num_envs, obs_dim)` 的 tensor）。
* `extras`：一个 **字典**，存放“不是主观测”的各种东西，包括：

  1. 其他 observation group（例如你定义的 `critic` 组 = 老师/privileged obs）
  2. 各种环境内部记录的额外信息（如 episode 统计、log 信息、time_outs 等，具体由 env / wrapper 决定）

RSL-RL 的 `RslRlVecEnvWrapper` 里有一段逻辑（伪代码）大概是这样的（你可以把这个当 mental model）：

```python
obs_dict = env.obs_buf  # 这里是一个 {group_name: tensor} 的 dict

# 选出主观测（一般是 "policy" 组），作为 RL 中的 obs
policy_obs = obs_dict["policy"]

extras = {}
# 其余组（比如 "critic"）被认为是 extra observations，放进 extras
extra_obs_groups = {k: v for k, v in obs_dict.items() if k != "policy"}
if extra_obs_groups:
    extras["observations"] = extra_obs_groups

# 此外还会把 env.extras 里的内容合并进来（如 log、time_outs 等）
extras.update(env.extras)

return policy_obs, extras
```

所以：**`extras` 里既可能有你的“老师观测”，也可能有 logging / time_out 标记等其它东西。**

---

## 2. 就你这份 `RobotEnvCfg`，哪些东西会进到 `extras`？

看你的 `ObservationsCfg`：

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        ...
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        ...
    critic: CriticCfg = CriticCfg()
```

也就是说：

* env 内部的 **observation_manager.compute()** 会得到一个 **字典**：

  ```python
  {
      "policy": <tensor of shape (num_envs, policy_dim)>,
      "critic": <tensor of shape (num_envs, critic_dim)>
  }
  ```
* 在 `ManagerBasedRLEnv.step()` / `get_observations()` 返回给 RSL-RL wrapper 时：

  * **主观测**：只取 `"policy"` 这一组（给 actor / 学生）
  * 其它组（这里只有 `"critic"`）会被 wrapper 视为 “extra observations”，放进 `extras`，形式大概类似：

    ```python
    extras = {
        "observations": {
            "critic": critic_obs_tensor,  # (num_envs, critic_dim)
        },
        # 还可能有 "log" / "time_outs" 等其它键
    }
    ```

再加上 `ManagerBasedEnv` / `ManagerBasedRLEnv` 自己维护的 `self.extras` 字典，里面通常会在：

* reset / step 时被各个 manager 写一些信息：

  * 例如 `reward_manager` 写 episode return
  * `termination_manager` 写哪些 env 触发了哪类终止
  * 你手动往里塞的一些 debug metric

RSL-RL 的 wrapper 在 `step()` 时还会把 `truncated`（time_out）也放到 `extras["time_outs"]` 里（用于无限/有限 horizon 的区分）。([docs.robotsfan.com][1])

**简单记忆：**

> 对你这个 env 来说，`obs` ≈ policy 组（学生），
> `extras["observations"]["critic"]` ≈ critic 组（老师 + privileged obs），
> 其它诸如 `extras["time_outs"]`、`extras["log"]` 是环境的附加信息。

你可以在代码里直接验证一下（强烈建议）：

```python
obs, extras = self.env.get_observations()
print(type(obs), obs.shape)             # (num_envs, policy_dim)
print(extras.keys())                    # 看看有啥键
print(extras.get("observations", {}).keys())  # 应该有 "critic"
print(extras["observations"]["critic"].shape) # (num_envs, critic_dim)
```

---

## 3. 想分开学生 / 教师观测，应该怎么做？

你现在的配置其实已经是 **经典 student / teacher（actor / critic privileged obs）写法** 了，接下来只要在训练代码里正确取就行。

### 3.1 “学生 / 教师”在这个配置里的对应关系

* 学生（policy / actor）：用 `ObservationsCfg.PolicyCfg` 对应的观测

  * 你已经把 `PolicyCfg.enable_corruption = True`，可以在这里做噪声 / 不完全观测等处理，适合作为 **学生观测**。
* 老师（critic / privileged）：用 `ObservationsCfg.CriticCfg` 对应的观测

  * 可以包含更多的、甚至是“作弊”的信息（例如真实速度、traj、高度图等），不对真实机器人暴露，只给 critic / teacher 模块用。

这就是 IsaacLab 官方 legged 示例默认采用的“学生 / 老师分观测”的套路。

### 3.2 在训练代码中怎么取？

#### 情形 A：你用的是 RSL-RL + `RslRlVecEnvWrapper`

典型 step / get_observations 写法：

```python
# 1) 取观测
obs, extras = env.get_observations()   # obs: 学生; extras: 里面藏着老师
student_obs = obs                      # shape: (num_envs, policy_dim)

# 2) 从 extras 里取老师（critic）观测
teacher_obs = None
if "observations" in extras and "critic" in extras["observations"]:
    teacher_obs = extras["observations"]["critic"]  # shape: (num_envs, critic_dim)

# 3) 你就可以：
# - 用 student_obs 喂 actor 网络
# - 用 teacher_obs 喂 critic 网络 / teacher policy / AMP discriminator 等
```

如果你自己写 runner，而不是用官方 PPO runner，可以显式把 `teacher_obs` 传给 critic：

```python
value = critic(student_obs, teacher_obs)  # 或者只传 teacher_obs，看你怎么设计
```

#### 情形 B：你不想通过 `extras`，想自己手动控制

你也可以直接调用 env 的 observation_manager：

```python
# 不通过 wrapper，直接问 env 当前所有观测
obs_dict = env.unwrapped.observation_manager.compute(update_history=False)

student_obs = obs_dict["policy"]   # (num_envs, policy_dim)
teacher_obs = obs_dict["critic"]   # (num_envs, critic_dim)
```

这样就完全自己掌控，不依赖 wrapper 帮你塞进 `extras`。

---

## 4. 如果想更明确地区分并 debug 学生 / 老师观测？

给你几个实用操作：

### 4.1 打印 group 信息和 shape

```python
obs_mgr = env.unwrapped.observation_manager
print("groups:", obs_mgr.active_terms.keys())
for group_name in obs_mgr.active_terms:
    print(group_name,
          "dim =", obs_mgr.group_obs_dim[group_name],
          "concat =", obs_mgr.group_obs_concatenate[group_name])
```

你会看到类似：

* `policy dim = (N,) concat = True`
* `critic dim = (M,) concat = True`

### 4.2 在一步训练里实际 dump 一下

```python
obs, extras = env.get_observations()
print("student_obs:", obs.shape)
if "observations" in extras:
    for k, v in extras["observations"].items():
        print(f"extra obs group {k}: {v.shape}")
print("extras keys:", extras.keys())
```

跑几步后，你就非常直观地知道：

* 学生 obs 是什么 shape
* 老师/critic obs 是在哪里、是什么 shape
* `extras` 里还有哪些信息可以利用（如 episode 统计、time_outs）

---

## 5. 一句话总结

* **`extras` = “除了主观测以外，我这个 vectorized env 想顺带告诉你的所有东西”的垃圾桶**——包括别的 obs 组（比如 `critic`/老师）、log、time_outs 等。
* 你的配置已经天然把 **学生** 映射到 `policy` 组，把 **老师 / privileged** 映射到 `critic` 组。
* 用 `obs, extras = env.get_observations()` 时：

  * `obs` 给学生
  * 从 `extras["observations"]["critic"]` 拿老师，或者直接用 `observation_manager.compute()` 自己取。

如果你把你现在的训练脚本 core loop 贴一段出来，我还能帮你把“学生/教师观测流向”画成一个简洁的数据流图，你以后看就一目了然 👀

[1]: https://docs.robotsfan.com/isaaclab/_modules/isaaclab_rl/rsl_rl/vecenv_wrapper.html?utm_source=chatgpt.com "isaaclab_rl.rsl_rl.vecenv_wrapper — Isaac Lab 文档"
