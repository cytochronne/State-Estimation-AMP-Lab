import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyMonitorHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] | tuple[int, ...], dropout_rate: float = 0.2):
        super().__init__()
        dims = list(hidden_dims or [])
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for dim in dims:
            layers.append(nn.Linear(prev, int(dim)))
            layers.append(nn.Dropout(p=float(dropout_rate)))
            layers.append(nn.ELU())
            prev = int(dim)
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.mean_head = nn.Linear(prev, int(output_dim))
        self.logvar_head = nn.Linear(prev, int(output_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        mean = self.mean_head(z)
        log_var = self.logvar_head(z)
        return mean, log_var

    @staticmethod
    def nll_loss(y_gt: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.exp(-log_var) * torch.square(y_gt - mean) + 0.5 * log_var

    @torch.no_grad()
    def predict_uncertainty(self, x: torch.Tensor, num_passes: int = 10) -> dict:
        prev_mode = self.training
        self.train()
        means = []
        vars_ = []
        for _ in range(int(num_passes)):
            m, lv = self.forward(x)
            means.append(m)
            vars_.append(torch.exp(lv))
        self.train(prev_mode)
        mstack = torch.stack(means, dim=0)
        vstack = torch.stack(vars_, dim=0)
        model_var = (mstack.pow(2).mean(dim=0) - mstack.mean(dim=0).pow(2))
        data_var = vstack.mean(dim=0)
        return {"model_var": model_var, "data_var": data_var, "total_var": model_var + data_var}

