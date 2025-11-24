# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl_AEMP.modules import Discriminator, StudentTeacher, StudentTeacherRecurrent
from rsl_rl_AEMP.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        discriminator_cfg: dict | None = None,
        adv_loss_weight: float = 0.0,
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.discriminator = None
        self.discriminator_optimizer = None
        self.discriminator_updates = 1
        self.discriminator_grad_pen_lambda = 0.0
        self.adv_loss_weight = adv_loss_weight
        self.storage = None  # initialized later
        # only train the student encoder (including recurrent memory)
        self.encoder_parameters: list[nn.Parameter] = []
        if hasattr(self.policy, "memory_s"):
            self.encoder_parameters.extend(p for p in self.policy.memory_s.parameters())
        student_module = getattr(self.policy, "student", None)
        if student_module is not None and hasattr(student_module, "get"):
            encoder_module = student_module.get("encoder")
            if encoder_module is not None:
                self.encoder_parameters.extend(p for p in encoder_module.parameters())
        elif hasattr(self.policy, "student_encoder"):
            self.encoder_parameters.extend(p for p in self.policy.student_encoder.parameters())

        if not self.encoder_parameters:
            raise ValueError("No encoder parameters found to optimise. Ensure the policy exposes an encoder module.")

        # freeze other student components if present
        policy_head = getattr(self.policy, "student_policy_head", None)
        if policy_head is not None:
            for param in policy_head.parameters():
                param.requires_grad_(False)

        self.optimizer = optim.Adam(self.encoder_parameters, lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # # initialize the loss function
        # if loss_type == "mse":
        #     self.loss_fn = nn.functional.mse_loss
        # elif loss_type == "huber":
        #     self.loss_fn = nn.functional.huber_loss
        # else:
        #     raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

        if discriminator_cfg is not None:
            self._build_discriminator(discriminator_cfg, learning_rate)

    def _build_discriminator(self, cfg: dict, default_lr: float) -> None:
        cfg = dict(cfg)

        hidden_dims = cfg.get("hidden_layer_sizes", [256, 256])
        input_dim = cfg.get("input_dim", getattr(self.policy, "teacher_latent_dim", None))
        if input_dim is None:
            raise ValueError(
                "Could not infer discriminator input dimension. Please provide 'input_dim' in discriminator_cfg "
                "or ensure the policy exposes 'teacher_latent_dim'."
            )

        loss_type = cfg.get("loss_type", "BCEWithLogits")
        use_minibatch_std = cfg.get("use_minibatch_std", True)
        eta_wgan = cfg.get("eta_wgan", 0.3)

        self.discriminator = Discriminator(
            input_dim=input_dim,
            hidden_layer_sizes=hidden_dims,
            device=self.device,
            loss_type=loss_type,
            eta_wgan=eta_wgan,
            use_minibatch_std=use_minibatch_std,
        )
        disc_lr = cfg.get("learning_rate", default_lr)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=disc_lr)
        self.discriminator_updates = int(cfg.get("updates_per_step", 1))
        self.discriminator_grad_pen_lambda = float(cfg.get("grad_penalty_lambda", 0.0))
        self.discriminator.train()

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )
    
    def act(self, obs, teacher_obs):
        # use the teacher to interact with the environment
        teacher_actions = self.policy.evaluate(teacher_obs).detach()
        self.transition.actions = teacher_actions
        self.transition.privileged_actions = teacher_actions
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_adv_loss = 0.0
        mean_disc_loss = 0.0
        mean_student_score = 0.0
        mean_teacher_score = 0.0
        # count how many generator backward passes accumulated since last step
        accum_cnt = 0
        adv_cnt = 0
        disc_cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            if self.discriminator is not None:
                self.discriminator.train()
            for obs, privileged_obs, _, privileged_actions, dones in self.storage.generator():

                # compute student and teacher latents
                
                student_latent = self.policy.get_student_latent(obs)
                total_loss = 0.0

                if self.discriminator is not None:
                    with torch.no_grad():
                        teacher_latent = self.policy.evaluate_feature(privileged_obs)
                        # Log scores
                        mean_student_score += self.discriminator.classify(student_latent).mean().item()
                        mean_teacher_score += self.discriminator.classify(teacher_latent).mean().item()

                    disc_loss_value = 0.0
                    for _ in range(self.discriminator_updates):
                        self.discriminator_optimizer.zero_grad()
                        disc_loss = self.discriminator.discriminator_loss(
                            student_latent.detach(), teacher_latent
                        )
                        grad_pen = self.discriminator.compute_grad_pen(
                            teacher_latent, student_latent.detach(), self.discriminator_grad_pen_lambda
                        )
                        disc_total = disc_loss + grad_pen
                        disc_total.backward()
                        if self.is_multi_gpu:
                            self._reduce_module_gradients(self.discriminator)
                        self.discriminator_optimizer.step()
                        disc_loss_value += disc_total.item()

                    disc_loss_value /= max(self.discriminator_updates, 1)
                    mean_disc_loss += disc_loss_value
                    disc_cnt += 1

                    if self.adv_loss_weight != 0.0:
                        adv_loss = self.discriminator.generator_loss(student_latent)
                        total_loss = total_loss + self.adv_loss_weight * adv_loss
                        mean_adv_loss += adv_loss.item()
                        adv_cnt += 1

                # perform immediate backward for generator loss to avoid stale
                # references to discriminator parameters across its updates
                if torch.is_tensor(total_loss):
                    if accum_cnt == 0:
                        self.optimizer.zero_grad()
                    next_accum_cnt = accum_cnt + 1
                    retain_graph = (
                        self.gradient_length is not None
                        and self.gradient_length > 1
                        and (next_accum_cnt % self.gradient_length) != 0
                    )
                    total_loss.backward(retain_graph=retain_graph)
                    accum_cnt = next_accum_cnt
                    # perform optimizer step based on gradient length
                    if accum_cnt % self.gradient_length == 0:
                        if self.is_multi_gpu:
                            self.reduce_parameters(self.encoder_parameters)
                        if self.max_grad_norm:
                            nn.utils.clip_grad_norm_(self.encoder_parameters, self.max_grad_norm)
                        self.optimizer.step()
                        self.policy.detach_hidden_states()
                        accum_cnt = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        # flush remaining accumulated gradients
        if accum_cnt > 0:
            if self.is_multi_gpu:
                self.reduce_parameters(self.encoder_parameters)
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.encoder_parameters, self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()

        if self.discriminator is not None and adv_cnt > 0:
            mean_adv_loss /= adv_cnt
        if self.discriminator is not None and disc_cnt > 0:
            mean_disc_loss /= disc_cnt
            mean_student_score /= disc_cnt
            mean_teacher_score /= disc_cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {}
        if self.discriminator is not None:
            if adv_cnt > 0:
                loss_dict["adv"] = mean_adv_loss
            if disc_cnt > 0:
                loss_dict["disc"] = mean_disc_loss
                loss_dict["student_score"] = mean_student_score
                loss_dict["teacher_score"] = mean_teacher_score

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.discriminator is not None:
            model_params.append(self.discriminator.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.discriminator is not None and len(model_params) > 1:
            self.discriminator.load_state_dict(model_params[1])

    def reduce_parameters(self, params=None):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        if not self.is_multi_gpu:
            return
        if params is None:
            params = self.policy.parameters()
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel

    def _reduce_module_gradients(self, module: nn.Module) -> None:
        if not self.is_multi_gpu:
            return

        grads = [param.grad.view(-1) for param in module.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in module.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
