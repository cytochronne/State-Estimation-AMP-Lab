# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# rsl-rl
from rsl_rl_MLE_stage2.modules import Discriminator, StudentTeacher, StudentTeacherRecurrent
from rsl_rl_MLE_stage2.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        bc_loss_coef=1.0,
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
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
        self.storage = None  # initialized later
        
        # Collect all student parameters (encoder + policy head)
        self.student_parameters: list[nn.Parameter] = []
        
        # 1. Encoder parameters
        if hasattr(self.policy, "memory_s"):
            self.student_parameters.extend(p for p in self.policy.memory_s.parameters())
        student_module = getattr(self.policy, "student", None)
        if student_module is not None and hasattr(student_module, "get"):
            encoder_module = student_module.get("encoder")
            if encoder_module is not None:
                self.student_parameters.extend(p for p in encoder_module.parameters())
        elif hasattr(self.policy, "student_encoder"):
            self.student_parameters.extend(p for p in self.policy.student_encoder.parameters())

        # 2. Policy head parameters
        policy_head = getattr(self.policy, "student_policy_head", None)
        if policy_head is not None:
            # Ensure they are trainable
            for param in policy_head.parameters():
                param.requires_grad_(True)
            self.student_parameters.extend(p for p in policy_head.parameters())

        if not self.student_parameters:
            raise ValueError("No student parameters found to optimise.")

        self.optimizer = optim.Adam(self.student_parameters, lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO / RL parameters
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.bc_loss_coef = bc_loss_coef

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # Force "rl" type to get values/advantages buffers, but we modified RolloutStorage to also have privileged_actions
        self.storage = RolloutStorage(
            "rl",
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )
    
    def act(self, obs, teacher_obs):
        # 1. Student acts (RL)
        # Compute student actions and values
        # Note: We use teacher's critic for value estimation (Asymmetric Actor-Critic)
        # or if student had a critic we would use it. Here we assume teacher critic.
        
        # Student action
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # Value from Teacher (Critic)
        # We assume self.policy.teacher has evaluate() or similar.
        # If self.policy is TerrainAwareStudentTeacher, it has self.teacher.
        
        # TerrainAwareStudentTeacher
        self.transition.values = self.policy.teacher.evaluate(teacher_obs).detach()
        

        # 2. Teacher acts (for BC target)
        # We need teacher's action for BC loss
        if hasattr(self.policy, "teacher"):
             self.transition.privileged_actions = self.policy.teacher.act_inference(teacher_obs).detach()
        else:
             self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach() # This might be wrong if evaluate returns value

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
        mean_mle_loss = 0.0
        mean_bc_loss = 0.0
        mean_surrogate_loss = 0.0
        
        # Latent statistics
        mean_student_mean_norm = 0.0
        mean_student_sigma_mean = 0.0
        mean_teacher_mean_norm = 0.0
        
        gen_cnt = 0

        # Compute returns and advantages
        last_values = torch.zeros(self.storage.num_envs, 1, device=self.device)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

        # Generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, privileged_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch, \
            privileged_actions_batch in generator:

                # 1. Compute Latent (Encoder)
                student_mean, student_var = self.policy.get_student_latent(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
                
                # 2. MLE Loss (Encoder update only)
                with torch.no_grad():
                    teacher_latent = self.policy.evaluate_feature(privileged_obs_batch)
                mle_loss = F.gaussian_nll_loss(student_mean, teacher_latent, student_var)

                # 3. Detach Latent for Policy Head (Stop gradient from RL/BC to Encoder)
                student_mean_detached = student_mean.detach()
                student_var_detached = student_var.detach()

                # 4. Run Policy Head (Head update only)
                # Reconstruct policy input as in TerrainAwareStudentTeacher.act
                uncertainty = student_var_detached.sum(dim=-1, keepdim=True)
                policy_input = torch.cat([student_mean_detached, uncertainty], dim=-1)
                self.policy.update_distribution(policy_input)

                # 5. Compute PPO/BC variables
                actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                mu_batch = self.policy.action_mean
                sigma_batch = self.policy.action_std
                entropy_batch = self.policy.entropy

                # 6. PPO Loss (Surrogate)
                # Adaptive LR / KL
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # 7. BC Loss (MSE)
                # Student action (mu_batch) vs Teacher action (privileged_actions_batch)
                bc_loss = F.mse_loss(mu_batch, privileged_actions_batch)

                # Total Loss
                # mle_loss affects encoder
                # surrogate_loss + bc_loss affects policy head (due to detach)
                loss = surrogate_loss + self.bc_loss_coef * bc_loss + mle_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.student_parameters, self.max_grad_norm)
                self.optimizer.step()

                # Logging
                mean_mle_loss += mle_loss.item()
                mean_bc_loss += bc_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                gen_cnt += 1
                
                # Debug stats
                with torch.no_grad():
                    mean_student_mean_norm += student_mean.norm(dim=-1).mean().item()
                    mean_student_sigma_mean += student_var.sqrt().mean().item()
                    mean_teacher_mean_norm += teacher_latent.norm(dim=-1).mean().item()
                    
                    # Cache last batch data for saving to disk later
                    self.last_debug_data = {
                        "obs": obs_batch.detach().cpu(),
                        "privileged_obs": privileged_obs_batch.detach().cpu(),
                        "student_mean": student_mean.detach().cpu(),
                        "student_sigma": student_var.sqrt().detach().cpu(),
                        "teacher_latent": teacher_latent.detach().cpu(),
                    }

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_mle_loss /= num_updates
        mean_bc_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_student_mean_norm /= num_updates
        mean_student_sigma_mean /= num_updates
        mean_teacher_mean_norm /= num_updates

        self.storage.clear()

        # --- Log Histograms to WandB (Rank 0 only) ---
        if self.gpu_global_rank == 0 and self.num_updates % 100 == 0:
            try:
                import wandb
                if wandb.run is not None and hasattr(self, "last_debug_data") and self.last_debug_data is not None:
                    hists = {
                        "latent_dist/student_mean": wandb.Histogram(self.last_debug_data["student_mean"].numpy()),
                        "latent_dist/student_sigma": wandb.Histogram(self.last_debug_data["student_sigma"].numpy()),
                        "latent_dist/teacher": wandb.Histogram(self.last_debug_data["teacher_latent"].numpy()),
                    }
                    wandb.log(hists, commit=False)
            except ImportError:
                pass
        # ---------------------------------------------

        # construct the loss dictionary
        loss_dict = {
            "mle_loss": mean_mle_loss,
            "bc_loss": mean_bc_loss,
            "surrogate_loss": mean_surrogate_loss,
            "latent/student_mean_norm": mean_student_mean_norm,
            "latent/student_uncertainty": mean_student_sigma_mean,
            "latent/teacher_mean_norm": mean_teacher_mean_norm,
        }

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self, params=None):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        if not self.is_multi_gpu:
            return
        if params is None:
            params = self.student_parameters
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
