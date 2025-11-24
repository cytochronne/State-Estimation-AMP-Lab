#!/usr/bin/env bash
# script: run_train_with_wandb.sh



# 设置 W&B API Key（请替换下面 YOUR_API_KEY）
export WANDB_API_KEY="55e33d6c7c19a3c852a65f91467588ead5036868"

# （可选）设置 W&B 项目／实体，如果你想指定
# export WANDB_PROJECT="your_project_name"
# export WANDB_ENTITY="your_team_or_username"

# 激活你对应的 conda 环境（如果需要）
# conda activate AMP

cd unitree_rl_lab/

# 运行教师训练命令
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Velocity --num_envs 2048 --device cuda:1 --log_root /home/dataset/yanzhe/SEAMPlog
# 运行学生encoder训练命令
# python scripts/rsl_rl/train_AEMP.py --headless --task Unitree-Go2-Velocity-Student-Encoder --num_envs 2048 --device cuda:2 --log_root /home/dataset/yanzhe/SEAMPlog/rsl_rl/unitree_go2_velocity/2025-11-11_15-51-23/model_19600.pt
# 脚本结束
