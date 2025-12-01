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
from rsl_rl_MLE.modules import Discriminator, StudentTeacher, StudentTeacherRecurrent
from rsl_rl_MLE.storage import RolloutStorage


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
        # PPO parameters
        clip_param=0.2,
        num_mini_batches=4,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        schedule="fixed",
        policy_head_lr=1e-3,
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
        self.storage = None  # initialized later
        # only train the student encoder (including recurrent memory)
        self.encoder_parameters: list[nn.Parameter] = []
        if hasattr(self.policy, "memory_s"):
            self.encoder_parameters.extend(p for p in self.policy.memory_s.parameters())
        
        # Try to find encoder parameters
        student_module = getattr(self.policy, "student", None)
        if hasattr(self.policy, "student_encoder"):
             self.encoder_parameters.extend(p for p in self.policy.student_encoder.parameters())
        elif student_module is not None and hasattr(student_module, "get"):
            encoder_module = student_module.get("encoder")
            if encoder_module is not None:
                self.encoder_parameters.extend(p for p in encoder_module.parameters())

        if not self.encoder_parameters:
            raise ValueError("No encoder parameters found to optimise. Ensure the policy exposes an encoder module.")

        # freeze other student components if present
        policy_head = getattr(self.policy, "student_policy_head", None)
        if policy_head is not None:
            for param in policy_head.parameters():
                param.requires_grad_(False)

        self.optimizer = optim.Adam(self.encoder_parameters, lr=learning_rate)
        
        # Setup Policy Head Optimizer for BC and RL
        self.policy_head_parameters = []
        if policy_head is not None:
            # Unfreeze for head training
            for param in policy_head.parameters():
                param.requires_grad_(True)
            self.policy_head_parameters.extend(policy_head.parameters())
            
        if not self.policy_head_parameters:
             print("Warning: No policy head parameters found. BC/RL updates will be skipped for head.")
             self.optimizer_head = None
        else:
             self.optimizer_head = optim.Adam(self.policy_head_parameters, lr=policy_head_lr)

        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.desired_kl = desired_kl
        self.schedule = schedule

        # # initialize the loss function
        # if loss_type == "mse":
        #     self.loss_fn = nn.functional.mse_loss
        # elif loss_type == "huber":
        #     self.loss_fn = nn.functional.huber_loss
        # else:
        #     raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            "distillation", # Use distillation type but RolloutStorage now allocates RL buffers too
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )
    
    def act(self, obs, teacher_obs):
        # Student interacts with environment
        self.transition.actions = self.policy.act(obs).detach()
        
        # Compute RL data
        if hasattr(self.policy, "get_actions_log_prob"):
            self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        elif hasattr(self.policy, "distribution") and self.policy.distribution is not None:
             self.transition.actions_log_prob = self.policy.distribution.log_prob(self.transition.actions).sum(dim=-1).unsqueeze(-1).detach()
        
        # Value function (if available, else 0)
        # Note: evaluate(teacher_obs) returns teacher actions in StudentTeacher!
        # We need a critic. If none, use 0.
        # Assuming StudentTeacher doesn't have a critic for now.
        self.transition.values = torch.zeros(self.transition.actions.shape[0], 1, device=self.device)
        
        if hasattr(self.policy, "action_mean"):
            self.transition.action_mean = self.policy.action_mean.detach()
        if hasattr(self.policy, "action_std"):
            self.transition.action_sigma = self.policy.action_std.detach()

        # Teacher actions for BC
        teacher_actions = self.policy.evaluate(teacher_obs).detach()
        self.transition.privileged_actions = teacher_actions
        
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def compute_returns(self, last_critic_obs):
        # Use 0 for last value if no critic
        last_values = torch.zeros(last_critic_obs.shape[0], 1, device=self.device)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

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
        mean_rl_loss = 0.0
        
        # Latent statistics
        mean_student_mean_norm = 0.0
        mean_student_sigma_mean = 0.0
        mean_teacher_mean_norm = 0.0
        
        accum_cnt = 0
        gen_cnt = 0

        for epoch in range(self.num_learning_epochs):
            # --- Generator Loop (MLE + BC) ---
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            for obs, privileged_obs, _, teacher_actions, dones in self.storage.generator():

                # compute student and teacher latents
                student_mean, student_var = self.policy.get_student_latent(obs)
                
                with torch.no_grad():
                    teacher_latent = self.policy.evaluate_feature(privileged_obs)

                # --- Statistics & Debug Data Collection ---
                with torch.no_grad():
                    gen_cnt += 1
                    mean_student_mean_norm += student_mean.norm(dim=-1).mean().item()
                    mean_student_sigma_mean += student_var.sqrt().mean().item()
                    mean_teacher_mean_norm += teacher_latent.norm(dim=-1).mean().item()

                    # Cache last batch data for saving to disk later
                    self.last_debug_data = {
                        "obs": obs.detach().cpu(),
                        "privileged_obs": privileged_obs.detach().cpu(),
                        "student_mean": student_mean.detach().cpu(),
                        "student_sigma": student_var.sqrt().detach().cpu(),
                        "teacher_latent": teacher_latent.detach().cpu(),
                    }
                # ------------------------------------------

                # MLE Loss
                mle_loss = F.gaussian_nll_loss(student_mean, teacher_latent, student_var)
                mean_mle_loss += mle_loss.item()
                
                # BC Loss
                # Detach latent to prevent gradients flowing to encoder
                student_latent_bc = student_mean.detach()
                if hasattr(self.policy, "student_policy_head"):
                    student_action_bc = self.policy.student_policy_head(student_latent_bc)
                    bc_loss = (student_action_bc - teacher_actions).pow(2).mean()
                    mean_bc_loss += bc_loss.item()
                else:
                    bc_loss = torch.tensor(0.0, device=self.device)

                # perform immediate backward
                if accum_cnt == 0:
                    self.optimizer.zero_grad()
                    if self.optimizer_head:
                        self.optimizer_head.zero_grad()
                
                next_accum_cnt = accum_cnt + 1
                retain_graph = (
                    self.gradient_length is not None
                    and self.gradient_length > 1
                    and (next_accum_cnt % self.gradient_length) != 0
                )
                
                # MLE Backward (Encoder)
                mle_loss.backward(retain_graph=retain_graph)
                
                # BC Backward (Head)
                if self.optimizer_head:
                    bc_loss.backward(retain_graph=retain_graph)

                accum_cnt = next_accum_cnt
                
                if accum_cnt % self.gradient_length == 0:
                    if self.is_multi_gpu:
                        self.reduce_parameters(self.encoder_parameters)
                        if self.optimizer_head:
                            self.reduce_parameters(self.policy_head_parameters)
                            
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.encoder_parameters, self.max_grad_norm)
                        if self.optimizer_head:
                            nn.utils.clip_grad_norm_(self.policy_head_parameters, self.max_grad_norm)
                            
                    self.optimizer.step()
                    if self.optimizer_head:
                        self.optimizer_head.step()
                        
                    self.policy.detach_hidden_states()
                    accum_cnt = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

            # flush remaining accumulated gradients
            if accum_cnt > 0:
                if self.is_multi_gpu:
                    self.reduce_parameters(self.encoder_parameters)
                    if self.optimizer_head:
                        self.reduce_parameters(self.policy_head_parameters)
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.encoder_parameters, self.max_grad_norm)
                    if self.optimizer_head:
                        nn.utils.clip_grad_norm_(self.policy_head_parameters, self.max_grad_norm)
                self.optimizer.step()
                if self.optimizer_head:
                    self.optimizer_head.step()
                self.policy.detach_hidden_states()
                accum_cnt = 0

            # --- RL Loop (Mini-batch) ---
            if self.optimizer_head:
                # Choose generator based on recurrence
                if getattr(self.policy, "is_recurrent", False):
                    generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, num_epochs=1)
                else:
                    generator = self.storage.mini_batch_generator(self.num_mini_batches, num_epochs=1)

                # Run 1 epoch of RL updates per outer epoch
                for batch in generator:
                    obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, rnd_state_batch = batch
                    
                    # PPO Update
                    if getattr(self.policy, "is_recurrent", False):
                        self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    else:
                        self.policy.act(obs_batch)
                    
                    if hasattr(self.policy, "get_actions_log_prob"):
                        actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
                    elif hasattr(self.policy, "distribution"):
                        actions_log_prob_batch = self.policy.distribution.log_prob(actions_batch).sum(dim=-1).unsqueeze(-1)
                    
                    # Value function (dummy 0)
                    value_batch = torch.zeros_like(target_values_batch)
                    
                    entropy_batch = self.policy.entropy

                    # Surrogate loss
                    ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                    surrogate = -torch.squeeze(advantages_batch) * ratio
                    surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                    # Value loss (dummy)
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                    
                    self.optimizer_head.zero_grad()
                    loss.backward()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy_head_parameters, self.max_grad_norm)
                    self.optimizer_head.step()
                    
                    mean_rl_loss += loss.item()

        if gen_cnt > 0:
            mean_mle_loss /= gen_cnt
            mean_bc_loss /= gen_cnt
        
        # Normalize RL loss by number of batches * epochs
        mean_rl_loss /= (self.num_learning_epochs * self.num_mini_batches)

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

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
        loss_dict = {}
        
        if gen_cnt > 0:
            loss_dict["mle_loss"] = mean_mle_loss
            loss_dict["bc_loss"] = mean_bc_loss
            loss_dict["rl_loss"] = mean_rl_loss
            loss_dict["latent/student_mean_norm"] = mean_student_mean_norm / gen_cnt
            loss_dict["latent/student_uncertainty"] = mean_student_sigma_mean / gen_cnt
            loss_dict["latent/teacher_mean_norm"] = mean_teacher_mean_norm / gen_cnt

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
