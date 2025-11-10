# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Sequence

from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class TerrainAwareRecurrentActorCritic(nn.Module):
    """Recurrent actor-critic that encodes terrain scans and temporal context separately.

    The observation is split into two parts:
        * Height scanner readings (last ``height_obs_dim`` elements) are passed through a dedicated MLP.
        * The remaining proprioceptive observations are processed by an RNN to capture history.
    The concatenated features are consumed by standard actor / critic heads.
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        height_obs_dim: int,
    actor_hidden_dims: Sequence[int] = (256, 256, 256),
    critic_hidden_dims: Sequence[int] = (256, 256, 256),
    height_encoder_dims: Sequence[int] | None = (256, 128),
    fusion_encoder_dims: Sequence[int] | None = (256, 256),
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "TerrainAwareRecurrentActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        if height_obs_dim < 0:
            raise ValueError("height_obs_dim must be non-negative.")

        if height_obs_dim == 0:
            self.actor_height_dim = 0
            self.critic_height_dim = 0
        elif height_obs_dim >= num_actor_obs or height_obs_dim >= num_critic_obs:
            print(
                "[WARN] Requested height_obs_dim exceeds available observations. "
                "Falling back to using all observations without a dedicated terrain encoder."
            )
            self.actor_height_dim = 0
            self.critic_height_dim = 0
        else:
            self.actor_height_dim = height_obs_dim
            self.critic_height_dim = height_obs_dim
        self.actor_core_dim = num_actor_obs - self.actor_height_dim
        self.critic_core_dim = num_critic_obs - self.critic_height_dim

        if self.actor_core_dim <= 0 or self.critic_core_dim <= 0:
            raise ValueError("Non-height observation dimensions must be positive.")

        activation_name = activation

        # Terrain encoder
        self.height_encoder, self.height_embedding_dim = self._build_height_encoder(
            self.actor_height_dim, height_encoder_dims, activation_name
        )

        # Recurrent memories for proprioceptive streams
        self.memory_actor = Memory(
            input_size=self.actor_core_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )
        self.memory_critic = Memory(
            input_size=self.critic_core_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )

        # Policy and value heads
        actor_fusion_in_dim = self.actor_core_dim + rnn_hidden_dim + self.height_embedding_dim
        critic_fusion_in_dim = self.critic_core_dim + rnn_hidden_dim + self.height_embedding_dim
        
        self.actor_fusion_encoder, self.actor_fusion_dim = self._build_fusion_encoder(
            actor_fusion_in_dim, fusion_encoder_dims, activation_name
        )
        self.critic_fusion_encoder, self.critic_fusion_dim = self._build_fusion_encoder(
            critic_fusion_in_dim, fusion_encoder_dims, activation_name
        )

        self.actor = self._build_head(self.actor_fusion_dim, actor_hidden_dims, num_actions, activation_name)
        self.critic = self._build_head(self.critic_fusion_dim, critic_hidden_dims, 1, activation_name)

        print(f"Terrain encoder MLP: {self.height_encoder}")
        print(f"Actor fusion encoder: {self.actor_fusion_encoder}")
        print(f"Critic fusion encoder: {self.critic_fusion_encoder}")
        print(f"Actor recurrent head: {self.actor}")
        print(f"Critic recurrent head: {self.critic}")

        # Action noise configuration
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("Unknown standard deviation type. Should be 'scalar' or 'log'.")

        Normal.set_default_validate_args(False)
        self.distribution = None

    @staticmethod
    def _build_head(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation_name: str) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)
    
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(resolve_nn_activation(activation_name))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_height_encoder(
        input_dim: int, hidden_dims: Sequence[int] | None, activation_name: str
    ) -> tuple[nn.Module, int]:
        if input_dim == 0 or not hidden_dims:
            return nn.Identity(), input_dim

        dims = list(hidden_dims)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for idx, dim in enumerate(dims):
            layers.append(nn.Linear(prev_dim, dim))
            if idx < len(dims) - 1:
                layers.append(resolve_nn_activation(activation_name))
            prev_dim = dim
        return nn.Sequential(*layers), dims[-1]

    @staticmethod
    def _build_fusion_encoder(
        input_dim: int, hidden_dims: Sequence[int] | None, activation_name: str
    ) -> tuple[nn.Module, int]:
        if input_dim == 0:
            return nn.Identity(), 0

        if not hidden_dims:
            return nn.Identity(), input_dim

        dims = list(hidden_dims)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for idx, dim in enumerate(dims):
            layers.append(nn.Linear(prev_dim, dim))
            if idx < len(dims) - 1:
                layers.append(resolve_nn_activation(activation_name))
            prev_dim = dim
        return nn.Sequential(*layers), dims[-1]

    def reset(self, dones=None):
        self.memory_actor.reset(dones)
        self.memory_critic.reset(dones)

    def get_hidden_states(self):
        return self.memory_actor.hidden_states, self.memory_critic.hidden_states

    def detach_hidden_states(self, dones=None):
        self.memory_actor.detach_hidden_states(dones)
        self.memory_critic.detach_hidden_states(dones)

    def _split_obs(self, obs: torch.Tensor, height_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if height_dim == 0:
            return obs, torch.empty(obs.shape[:-1] + (0,), device=obs.device, dtype=obs.dtype)
        return obs[..., :-height_dim], obs[..., -height_dim:]

    def _encode_height(self, height: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        if height.shape[-1] == 0:
            return height[..., :0]

        if masks is None:
            return self.height_encoder(height)

        time_steps, batch_size = height.shape[0], height.shape[1]
        height_flat = height.reshape(time_steps * batch_size, -1)
        encoded = self.height_encoder(height_flat)
        encoded = encoded.view(time_steps, batch_size, -1)
        encoded = unpad_trajectories(encoded, masks)
        return encoded.squeeze(0)

    def _prepare_features(
        self,
        observations: torch.Tensor,
        masks: torch.Tensor | None,
        hidden_states,
        memory: Memory,
        height_dim: int,
        fusion_encoder: nn.Module,
    ) -> torch.Tensor:
        core, height = self._split_obs(observations, height_dim)
        rnn_out = memory(core, masks, hidden_states).squeeze(0)
        height_feat = self._encode_height(height, masks)
        core_feat = self._encode_core(core, masks)

        features = []
        if core_feat.numel() != 0:
            features.append(core_feat)
        if rnn_out.numel() != 0:
            features.append(rnn_out)
        if height_feat.numel() != 0:
            features.append(height_feat)

        if not features:
            raise RuntimeError("No features available for fusion. Check observation configuration.")

        fusion_input = features[0] if len(features) == 1 else torch.cat(features, dim=-1)
        return fusion_encoder(fusion_input)

    @staticmethod
    def _encode_core(core: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        if core.shape[-1] == 0:
            return core[..., :0]
        if masks is None:
            return core
        encoded = unpad_trajectories(core, masks)
        return encoded.squeeze(0)

    def update_distribution(self, features: torch.Tensor) -> None:
        mean = self.actor(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError("Unknown standard deviation type. Should be 'scalar' or 'log'.")
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None):
        features = self._prepare_features(
            observations,
            masks,
            hidden_states,
            self.memory_actor,
            self.actor_height_dim,
            self.actor_fusion_encoder,
        )
        self.update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, observations):
        features = self._prepare_features(
            observations,
            None,
            None,
            self.memory_actor,
            self.actor_height_dim,
            self.actor_fusion_encoder,
        )
        return self.actor(features)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        features = self._prepare_features(
            critic_observations,
            masks,
            hidden_states,
            self.memory_critic,
            self.critic_height_dim,
            self.critic_fusion_encoder,
        )
        return self.critic(features)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True
