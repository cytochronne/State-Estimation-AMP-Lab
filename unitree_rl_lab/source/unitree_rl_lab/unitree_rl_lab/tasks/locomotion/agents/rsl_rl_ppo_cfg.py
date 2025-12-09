# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,RslRlDistillationAlgorithmCfg


@configclass
class TerrainAwarePpoActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "TerrainAwareActorCritic"
    height_obs_dim: int = 0
    height_encoder_dims = (256, 128)
    fusion_encoder_dims = (256, 128, 96)
    rnn_type = "lstm"
    rnn_hidden_dim = 256
    rnn_num_layers = 1
    noise_std_type = "scalar"
    uncertainty_method = "none"
    uncertainty_num_models = 1
    uncertainty_num_passes = 1
    uncertainty_dropout_prob = 0.0
    dz_hidden_dims = (128, 64)


@configclass
class TerrainAwareStudentTeacherCfg(RslRlPpoActorCriticCfg):
    class_name = "TerrainAwareStudentTeacher"
    teacher_height_obs_dim: int = 0
    student_height_obs_dim: int = 0
    height_encoder_dims = (256, 128)
    fusion_encoder_dims = (256, 128, 96)
    height_cnn_channels = (16, 32)
    rnn_type = "lstm"
    rnn_hidden_dim = 256
    rnn_num_layers = 1
    noise_std_type = "scalar"
    student_encoder_hidden_dims = (256, 256)
    student_policy_hidden_dims = (256, 256, 256)


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = ""  # same as task name
    empirical_normalization = False
    # obs_groups = {
    #     "policy": ["policy"],
    #     # optional: you may explicitly set critic; if omitted, resolve_obs_groups() will fill it
    #     # "critic": ["critic"],
    # }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # default uncertainty logging configuration
    uncertainty_cfg = {
        "method": "mc_dropout",
        "dropout_prob": 0.1,
        "num_passes": 10,
        "num_models": 1,
        "weight_noise_std": 0.0,
        "sample_size": 256,
    }


@configclass
class TerrainAwarePPORunnerCfg(BasePPORunnerCfg):
    policy = TerrainAwarePpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        height_obs_dim=187,
        uncertainty_method="mc_dropout",
        uncertainty_num_models=1,
        uncertainty_num_passes=10,
        uncertainty_dropout_prob=0.2,
        dz_hidden_dims=(128, 64),
    )


@configclass
class TerrainAwareDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name = "Distillation"
    learning_rate = 5.0e-4
    gradient_length = 15
    num_learning_epochs = 1
    max_grad_norm = 1.0
    #BCEWithLogits or Wasserstein
    loss_type = "BCEWithLogits"
    discriminator_cfg = {
        "hidden_layer_sizes": [256, 256],
        "learning_rate": 5.0e-4,
        "use_minibatch_std": False,
        "grad_penalty_lambda": 0.05,
    }
    adv_loss_weight = 1.0


@configclass
class TerrainAwareDistillationRunnerCfg(BasePPORunnerCfg):
    policy = TerrainAwareStudentTeacherCfg(
        init_noise_std=1.0,
        teacher_height_obs_dim=187,
        student_height_obs_dim=0,
        activation="elu",
    )
    algorithm = TerrainAwareDistillationAlgorithmCfg()
