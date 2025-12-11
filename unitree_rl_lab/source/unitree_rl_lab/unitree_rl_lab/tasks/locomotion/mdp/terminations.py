import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject


def root_height_below_terrain(env, minimum_clearance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w  # [N, R, 3]
    root_xy = asset.data.root_pos_w[:, :2].unsqueeze(1)
    dxy = torch.norm(hits[..., :2] - root_xy, dim=-1)  # [N, R]
    idx = torch.argmin(dxy, dim=1)  # [N]
    terrain_z = hits[torch.arange(hits.shape[0], device=hits.device), idx, 2]
    root_z = asset.data.root_pos_w[:, 2]
    return root_z < terrain_z + float(minimum_clearance)


def bad_orientation_adaptive(env, base_limit: float, slope_gain: float, max_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w
    centered = hits - torch.mean(hits, dim=1, keepdim=True)
    denom = float(max(hits.shape[1], 1))
    cov = torch.matmul(centered.transpose(1, 2), centered) / denom
    cov = 0.5 * (cov + cov.transpose(1, 2))
    diag = torch.diagonal(cov, dim1=1, dim2=2)
    scale = torch.mean(diag, dim=1, keepdim=True)
    eps = 1e-6 + 1e-3 * scale
    I = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype).unsqueeze(0)
    cov = cov + eps.unsqueeze(-1) * I
    try:
        eigvals, eigvecs = torch.linalg.eigh(cov)
        normal = eigvecs[:, :, 0]
    except RuntimeError:
        normal = torch.zeros((cov.shape[0], 3), device=cov.device, dtype=cov.dtype)
        normal[:, 2] = 1.0
    normal = torch.nn.functional.normalize(normal, dim=1)
    slope_angle = torch.acos(torch.clamp(normal[:, 2], -1.0, 1.0))
    limit = torch.clamp(base_limit + slope_gain * slope_angle, min=base_limit, max=max_limit)
    angle = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)).abs()
    return angle > limit
