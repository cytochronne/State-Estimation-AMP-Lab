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
        mean_mle_loss = 0.0
        
        # Latent statistics
        mean_student_mean_norm = 0.0
        mean_student_sigma_mean = 0.0
        mean_teacher_mean_norm = 0.0
        
        accum_cnt = 0
        gen_cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            
            for obs, privileged_obs, _, privileged_actions, dones in self.storage.generator():

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
                total_loss = mle_loss
                mean_mle_loss += mle_loss.item()

                # perform immediate backward
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

        if gen_cnt > 0:
            mean_mle_loss /= gen_cnt

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
