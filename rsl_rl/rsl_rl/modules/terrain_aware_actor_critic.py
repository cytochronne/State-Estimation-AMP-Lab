# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Sequence, Tuple

from rsl_rl.utils import resolve_nn_activation


class TerrainAwareActorCritic(nn.Module):
    """Recurrent actor-critic that encodes terrain scans and temporal context separately.

    The observation is split into two parts:
        * Height scanner readings (last ``height_obs_dim`` elements) are passed through a dedicated MLP.
        * The remaining proprioceptive observations are processed by an RNN to capture history.
    The concatenated features are consumed by standard actor / critic heads.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        height_obs_dim: int,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        fusion_encoder_dims: Sequence[int] | None = (256, 128, 96),
        height_cnn_channels: Sequence[int] = (16, 32),
        height_map_shape: Tuple[int, int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        height_encoder_dims: Sequence[int] | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        uncertainty_method: str = "none",
        uncertainty_num_models: int = 1,
        uncertainty_num_passes: int = 1,
        uncertainty_dropout_prob: float = 0.0,
        dz_hidden_dims: Sequence[int] = (128, 64),
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            print(
                "TerrainAwareActorCritic.__init__ got unexpected arguments, which will be ignored: "
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

        self.height_map_shape = self._resolve_height_map_shape(height_obs_dim, height_map_shape)
        self.height_encoder, self.height_embedding_dim = self._build_height_cnn(
            self.height_map_shape, height_cnn_channels, activation_name
        )

        actor_fusion_in_dim = self.actor_core_dim + self.height_embedding_dim
        critic_fusion_in_dim = self.critic_core_dim + self.height_embedding_dim

        self.uncertainty_method = uncertainty_method
        self.uncertainty_num_models = max(int(uncertainty_num_models), 1)
        self.uncertainty_num_passes = max(int(uncertainty_num_passes), 1)
        self.uncertainty_dropout_prob = float(uncertainty_dropout_prob)

        # policy fusion encoder (no dropout to keep training behavior stable)
        self.actor_fusion_encoder, self.actor_fusion_dim = self._build_fusion_encoder(
            actor_fusion_in_dim, fusion_encoder_dims, activation_name, 0.0, False
        )

        # uncertainty encoders (separate from policy path)
        if self.uncertainty_method == "ensemble" and self.uncertainty_num_models > 1:
            self.actor_fusion_ensemble = nn.ModuleList()
            for _ in range(self.uncertainty_num_models):
                enc, _ = self._build_fusion_encoder(
                    actor_fusion_in_dim, fusion_encoder_dims, activation_name, 0.0, False
                )
                self.actor_fusion_ensemble.append(enc)
        elif self.uncertainty_method == "mc_dropout":
            self.actor_fusion_encoder_uncertainty, _ = self._build_fusion_encoder(
                actor_fusion_in_dim, fusion_encoder_dims, activation_name, self.uncertainty_dropout_prob, True
            )

        self.critic_fusion_encoder, self.critic_fusion_dim = self._build_fusion_encoder(
            critic_fusion_in_dim, fusion_encoder_dims, activation_name, 0.0, False
        )

        self.actor = self._build_head(self.actor_fusion_dim, actor_hidden_dims, num_actions, activation_name)
        self.critic = self._build_head(self.critic_fusion_dim, critic_hidden_dims, 1, activation_name)

        print(f"Terrain encoder CNN: {self.height_encoder}")
        print(f"Actor fusion encoder: {self.actor_fusion_encoder}")
        print(f"Critic fusion encoder: {self.critic_fusion_encoder}")
        print(f"Actor head: {self.actor}")
        print(f"Critic head: {self.critic}")

        self.d_z = self._build_head(self.actor_fusion_dim, dz_hidden_dims, 1, activation_name)

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
    def _resolve_height_map_shape(height_dim: int, explicit_shape: Tuple[int, int] | None) -> Tuple[int, int]:
        if height_dim == 0:
            return 0, 0

        if explicit_shape is not None:
            if explicit_shape[0] * explicit_shape[1] != height_dim:
                raise ValueError(
                    "Provided height_map_shape does not match height_obs_dim. "
                    f"Got {explicit_shape[0]}x{explicit_shape[1]} != {height_dim}."
                )
            return explicit_shape

        factors: list[Tuple[int, int]] = []
        for h in range(1, int(math.sqrt(height_dim)) + 1):
            if height_dim % h == 0:
                factors.append((h, height_dim // h))

        if not factors:
            raise ValueError(f"Unable to factorize height_obs_dim={height_dim} into a 2D map shape.")

        # choose the pair with the smallest aspect ratio difference to keep the map near-square
        best_h, best_w = min(factors, key=lambda hw: abs(hw[0] - hw[1]))
        return best_h, best_w

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
    def _build_height_cnn(
        map_shape: Tuple[int, int],
        channels: Sequence[int],
        activation_name: str,
    ) -> tuple[nn.Module, int]:
        height, width = map_shape
        if height == 0 or width == 0:
            return nn.Identity(), 0

        layers: list[nn.Module] = []
        in_channels = 1
        current_h, current_w = height, width
        for idx, out_channels in enumerate(channels):
            stride = 2 if current_h >= 4 and current_w >= 4 else 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
            layers.append(resolve_nn_activation(activation_name))
            in_channels = out_channels
            current_h = math.floor((current_h + 2 - 3) / stride + 1)
            current_w = math.floor((current_w + 2 - 3) / stride + 1)
        layers.append(nn.Flatten())
        embedding_dim = in_channels * max(current_h, 1) * max(current_w, 1)
        return nn.Sequential(*layers), embedding_dim

    @staticmethod
    def _build_fusion_encoder(
        input_dim: int,
        hidden_dims: Sequence[int] | None,
        activation_name: str,
        dropout_prob: float = 0.0,
        mc_dropout: bool = False,
    ) -> tuple[nn.Module, int]:
        if input_dim == 0:
            return nn.Identity(), 0

        if not hidden_dims:
            return nn.Identity(), input_dim

        class _MCDropout(nn.Module):
            def __init__(self, p: float):
                super().__init__()
                self.p = float(p)
            def forward(self, x):
                return F.dropout(x, self.p, True, False)

        dims = list(hidden_dims)
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for idx, dim in enumerate(dims):
            layers.append(nn.Linear(prev_dim, dim))
            if idx < len(dims) - 1:
                layers.append(resolve_nn_activation(activation_name))
                if mc_dropout and dropout_prob > 0.0:
                    layers.append(_MCDropout(dropout_prob))
            prev_dim = dim
        return nn.Sequential(*layers), dims[-1]

    def reset(self, dones=None):
        return None

    def get_hidden_states(self):
        return None, None

    def detach_hidden_states(self, dones=None):
        return None

    def _split_obs(self, obs: torch.Tensor, height_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if height_dim == 0:
            return obs, torch.empty(obs.shape[:-1] + (0,), device=obs.device, dtype=obs.dtype)
        return obs[..., :-height_dim], obs[..., -height_dim:]

    def _encode_height(self, height: torch.Tensor, height_dim: int) -> torch.Tensor:
        if height_dim == 0 or height.shape[-1] == 0:
            return height[..., :0]

        batch_shape = height.shape[:-1]
        height_flat = height.view(-1, height_dim)
        height_map = height_flat.view(-1, 1, *self.height_map_shape)
        encoded = self.height_encoder(height_map)
        return encoded.view(*batch_shape, -1)

    def _prepare_features(self, observations: torch.Tensor, height_dim: int, fusion_encoder: nn.Module) -> torch.Tensor:
        core, height = self._split_obs(observations, height_dim)
        height_feat = self._encode_height(height, height_dim)
        if height_feat.numel() != 0:
            fusion_input = torch.cat((core, height_feat), dim=-1)
        else:
            fusion_input = core
        return fusion_encoder(fusion_input)

    def update_distribution(self, features: torch.Tensor) -> None:
        mean = self.actor(features)
        if self.noise_std_type == "scalar":
            raw = self.std  # nn.Parameter(...), shape maybe [action_dim] or scalar
            # 把 raw 转到 mean 的 device/dtype
            raw = raw.to(device=mean.device, dtype=mean.dtype)
            std_unclamped = F.softplus(raw)  # > 0
            std = (std_unclamped + 1e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError("Unknown standard deviation type. Should be 'scalar' or 'log'.")
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None):
        features = self._prepare_features(observations, self.actor_height_dim, self.actor_fusion_encoder)
        self.update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, observations):
        features = self._prepare_features(observations, self.actor_height_dim, self.actor_fusion_encoder)
        return self.actor(features)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        features = self._prepare_features(critic_observations, self.critic_height_dim, self.critic_fusion_encoder)
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

    def score_dz(self, observations):
        core, height = self._split_obs(observations, self.actor_height_dim)
        height_feat = self._encode_height(height, self.actor_height_dim)
        if height_feat.numel() != 0:
            fusion_input = torch.cat((core, height_feat), dim=-1)
        else:
            fusion_input = core
        scores = []
        if self.uncertainty_method == "ensemble" and hasattr(self, "actor_fusion_ensemble"):
            for enc in self.actor_fusion_ensemble:
                z = enc(fusion_input)
                s = self.d_z(z).squeeze(-1)
                scores.append(s)
        elif self.uncertainty_method == "mc_dropout" and hasattr(self, "actor_fusion_encoder_uncertainty"):
            for _ in range(self.uncertainty_num_passes):
                z = self.actor_fusion_encoder_uncertainty(fusion_input)
                s = self.d_z(z).squeeze(-1)
                scores.append(s)
        else:
            z = self.actor_fusion_encoder(fusion_input)
            s = self.d_z(z).squeeze(-1)
            scores.append(s)
        stacked = torch.stack(scores, dim=0)
        var = stacked.var(dim=0, unbiased=False)
        mean = stacked.mean(dim=0)
        return mean, var, stacked
