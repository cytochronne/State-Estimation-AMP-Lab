#!/bin/bash
# ----------------------------------------------------------------------
# run_sim.sh - 适用于 Isaac Lab 和 Unitree RL Lab 的统一环境启动脚本
# 请放到 IsaacLab 根目录执行
# ----------------------------------------------------------------------

# 1. 激活 Conda 环境 (兼容 Zsh/Bash 的激活方式)
CONDA_BASE=$(conda info --base)
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate env_isaaclab

# 2. 获取 Isaac Sim 的绝对真实路径 (依赖于在 IsaacLab 根目录执行)
REAL_ISAAC_PATH=$(readlink -f _isaac_sim)

# 3. --- Python 模块路径配置 ---
PYTHONPATH_VAL=""
# 3a. 【IsaacLab 源码路径】 确保 isaaclab 本身能被找到
PYTHONPATH_VAL+="/home/cft/zikang/State-Estimation-AMP-Lab/IsaacLab/source/isaaclab/isaaclab:"
# 3b. 【Kit Core Python Path】 (包含 carb Python 模块)
PYTHONPATH_VAL+="$REAL_ISAAC_PATH/kit/kernel/py:"
# 3c. 【Isaac Sim 特定扩展】
PYTHONPATH_VAL+="$REAL_ISAAC_PATH/exts/isaacsim.simulation_app:"
PYTHONPATH_VAL+="$REAL_ISAAC_PATH/exts/omni.isaac.core:"
PYTHONPATH_VAL+="$REAL_ISAAC_PATH/exts/omni.isaac.kit"
export PYTHONPATH="$PYTHONPATH_VAL"

# 4. --- 原生库路径配置 (LD_LIBRARY_PATH) ---
# Kit Binaries 路径 (包含 libcarb.so)
export LD_LIBRARY_PATH="$REAL_ISAAC_PATH/kit:$LD_LIBRARY_PATH"

# 5. 运行 Isaac Sim 原生的环境设置脚本
. ./_isaac_sim/setup_conda_env.sh

# 6. 【关键修改】切换到目标项目目录
echo "Switching directory to unitree_rl_lab..."
cd /home/cft/zikang/State-Estimation-AMP-Lab/unitree_rl_lab

# 7. 执行 Unitree RL Lab 的 play 脚本
# python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity
# python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity-Student-Encoder
python scripts/rsl_rl/play_distill.py --task Unitree-Go2-Velocity-Student-Encoder --num_envs 1

echo "Script execution finished."