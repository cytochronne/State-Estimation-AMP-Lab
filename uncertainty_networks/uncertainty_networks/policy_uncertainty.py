import torch
import torch.nn as nn

from .uncertainty_networks import UncertaintyMLP


class PolicyUncertaintyEstimator:
    def __init__(
        self,
        model: UncertaintyMLP,
        dropout_prob: float,
        num_passes: int,
        num_models: int,
        method: str,
        weight_noise_std: float,
        device: str,
    ):
        self._model = model
        self.dropout_prob = dropout_prob
        self.num_passes = num_passes
        self.num_models = num_models
        self.method = method
        self.weight_noise_std = float(weight_noise_std)
        self.device = device

    @staticmethod
    def _extract_actor_spec(actor: nn.Sequential):
        linears = [m for m in actor if isinstance(m, nn.Linear)]
        if len(linears) == 0:
            raise RuntimeError("Actor head does not contain Linear layers; unsupported for uncertainty estimator")
        input_size = linears[0].in_features
        hidden_sizes = [layer.out_features for layer in linears[:-1]]
        output_size = linears[-1].out_features
        # infer activation from non-linear modules between linears; fallback to ELU
        activation_cls = nn.ELU
        for m in actor:
            if isinstance(m, nn.Linear):
                continue
            if isinstance(m, nn.Module):
                activation_cls = m.__class__
                break
        return input_size, hidden_sizes, output_size, activation_cls

    @classmethod
    def from_actor(
        cls,
        actor: nn.Sequential,
        dropout_prob: float = 0.1,
        num_passes: int = 10,
        num_models: int = 1,
        method: str = "mc_dropout",
        weight_noise_std: float = 0.0,
        device: str = "cpu",
    ):
        input_size, hidden_sizes, output_size, activation_cls = cls._extract_actor_spec(actor)
        # guard against zero hidden layers
        hidden_sizes = list(hidden_sizes)
        if len(hidden_sizes) == 0:
            hidden_sizes = [output_size]
        model = UncertaintyMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_prob=dropout_prob if method in ("mc_dropout", "mc_ensemble") else 0.0,
            num_passes=num_passes,
            num_models=num_models,
            initialization="rl",
            activation=activation_cls,
            device=device,
        )
        est = cls(model, dropout_prob, num_passes, num_models, method, weight_noise_std, device)
        est.sync_from_actor(actor)
        if est.method in ("ensemble", "mc_ensemble") and est.weight_noise_std > 0.0:
            est._apply_weight_noise()
        return est

    def sync_from_actor(self, actor: nn.Sequential):
        actor_linears = [m for m in actor if isinstance(m, nn.Linear)]
        for model in self._model._models:
            model_linears = [m for m in model if isinstance(m, nn.Linear)]
            for src, dst in zip(actor_linears, model_linears):
                dst.weight.data.copy_(src.weight.data)
                dst.bias.data.copy_(src.bias.data)

    def _apply_weight_noise(self):
        std = self.weight_noise_std
        if std <= 0:
            return
        for model in self._model._models:
            for layer in model:
                if isinstance(layer, nn.Linear):
                    noise_w = torch.randn_like(layer.weight) * std
                    noise_b = torch.randn_like(layer.bias) * std
                    layer.weight.data.add_(noise_w)
                    layer.bias.data.add_(noise_b)

    @torch.no_grad()
    def metrics(self, observations: torch.Tensor, sample_size: int | None = None):
        if sample_size is not None:
            observations = observations[:sample_size]
        preds = self._model(observations, shared_input=True)
        mean = preds.mean(dim=0)
        var = preds.var(dim=0, unbiased=False)
        mean_variance = var.mean().item()
        max_variance = var.max().item()
        per_action_mean_variance = var.mean(dim=0).detach().cpu().numpy().tolist()
        return {
            "mean_variance": mean_variance,
            "max_variance": max_variance,
            "per_action_mean_variance": per_action_mean_variance,
        }
