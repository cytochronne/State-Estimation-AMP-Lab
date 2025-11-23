"""Discriminator network used for adversarial encoder pre-training.

This module classifies whether a latent feature vector originates from the fixed
teacher encoder or the student encoder. It provides:

* a discriminator loss used to update itself so that teacher features map to the
  positive class and student features to the negative class;
* a generator loss used to guide the student encoder towards the teacher
  manifold by maximising the discriminator's confidence that student features
  are teacher-like.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F


class Discriminator(nn.Module):
    """Binary classifier operating on latent feature vectors."""

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        *,
        device: str = "cpu",
        loss_type: str = "BCEWithLogits",
        eta_wgan: float = 0.3,
        use_minibatch_std: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("`input_dim` must be a positive integer.")
        if not hidden_layer_sizes:
            raise ValueError("`hidden_layer_sizes` must contain at least one entry.")

        self.device = device
        self.use_minibatch_std = use_minibatch_std
        self.loss_type = loss_type if loss_type is not None else "BCEWithLogits"
        self.eta_wgan = eta_wgan

        layers: list[nn.Module] = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        final_in_dim = hidden_layer_sizes[-1] + (1 if use_minibatch_std else 0)
        self.linear = nn.Linear(final_in_dim, 1)

        self.to(self.device)

        if self.loss_type not in {"BCEWithLogits", "Wasserstein"}:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}. Supported types are 'BCEWithLogits' and 'Wasserstein'."
            )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return discriminator logits for the provided latent vectors."""

        h = self.trunk(latent)
        if self.use_minibatch_std:
            std_feat = self._minibatch_std_scalar(h)
            h = torch.cat([h, std_feat], dim=-1)
        return self.linear(h)

    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        """Return probabilities of the latent belonging to the teacher distribution."""

        logits = self.forward(latent)
        return torch.sigmoid(logits)

    def generator_loss(self, student_latent: torch.Tensor) -> torch.Tensor:
        """Loss for the student encoder (generator) to fool the discriminator."""

        logits = self.forward(student_latent)
        if self.loss_type == "BCEWithLogits":
            targets = torch.ones_like(logits, device=logits.device)
            return F.binary_cross_entropy_with_logits(logits, targets)

        logits = torch.tanh(self.eta_wgan * logits)
        return -logits.mean()

    def discriminator_loss(self, student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> torch.Tensor:
        """Loss for the discriminator to distinguish student from teacher latents."""

        student_logits = self.forward(student_latent)
        teacher_logits = self.forward(teacher_latent)

        if self.loss_type == "BCEWithLogits":
            student_targets = torch.zeros_like(student_logits, device=student_logits.device)
            teacher_targets = torch.ones_like(teacher_logits, device=teacher_logits.device)
            loss_student = F.binary_cross_entropy_with_logits(student_logits, student_targets)
            loss_teacher = F.binary_cross_entropy_with_logits(teacher_logits, teacher_targets)
            return 0.5 * (loss_student + loss_teacher)

        student_scores = torch.tanh(self.eta_wgan * student_logits)
        teacher_scores = torch.tanh(self.eta_wgan * teacher_logits)
        return student_scores.mean() - teacher_scores.mean()

    def compute_grad_pen(
        self,
        teacher_latent: torch.Tensor,
        student_latent: torch.Tensor,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """Gradient penalty (WGAN-GP or R1) for stabilising discriminator training."""

        if lambda_ <= 0.0:
            return torch.zeros((), device=self.device)

        if self.loss_type == "Wasserstein":
            alpha = torch.rand(teacher_latent.size(0), 1, device=teacher_latent.device)
            alpha = alpha.expand_as(teacher_latent)
            interpolates = alpha * teacher_latent + (1 - alpha) * student_latent
            interpolates = interpolates.detach().requires_grad_(True)
            logits = self.forward(interpolates)
            gradients = autograd.grad(
                outputs=logits,
                inputs=interpolates,
                grad_outputs=torch.ones_like(logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return lambda_ * (gradients.norm(2, dim=1) - 1.0).pow(2).mean()

        real = teacher_latent.detach().requires_grad_(True)
        logits = self.forward(real)
        gradients = autograd.grad(
            outputs=logits.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return 0.5 * lambda_ * gradients.pow(2).sum(dim=1).mean()

    def _minibatch_std_scalar(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[0] <= 1:
            return h.new_zeros((h.shape[0], 1))
        h_float = h.float()
        # add epsilon to avoid zero std which would trigger NaNs during backward
        variance = h_float.var(dim=0, unbiased=False)
        std = torch.sqrt(variance + 1.0e-6)
        std_mean = std.mean()
        return std_mean.expand(h.shape[0], 1).to(h.dtype)