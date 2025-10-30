import numpy as np
import torch
import torch.nn.functional as F


def compute_geometry_mask_and_weight(
    ref_cam,
    nearest_cam,
    ref_depth,              # shape [1, H, W]
    nearest_depth,          # shape [1, H, W]
    gaussians,
    pixel_noise_th: float,
    wo_use_geo_occ_aware: bool = False,
    geo_weight: float = 0.0,
):

    device = ref_depth.device

    H, W = ref_depth.squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(ref_depth.device)

    pts = gaussians.get_points_from_depth(ref_cam, ref_depth)
    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[3, :3]
    map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_depth,
                                                            pts_in_nearest_cam)

    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
    R = torch.tensor(nearest_cam.R).float().cuda()
    T = torch.tensor(nearest_cam.T).float().cuda()
    pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
    pts_in_view_cam = pts_ @ ref_cam.world_view_transform[:3, :3] + ref_cam.world_view_transform[3, :3]
    pts_projections = torch.stack(
        [pts_in_view_cam[:, 0] * ref_cam.Fx / pts_in_view_cam[:, 2] + ref_cam.Cx,
         pts_in_view_cam[:, 1] * ref_cam.Fy / pts_in_view_cam[:, 2] + ref_cam.Cy], -1).float()
    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)

    if not wo_use_geo_occ_aware:
        d_mask = d_mask & (pixel_noise < pixel_noise_th)

    if not wo_use_geo_occ_aware:
        weights = (1.0 / torch.exp(pixel_noise)).detach()
        weights[~d_mask] = 0
    else:
        weights = torch.ones_like(pixel_noise, device=device)
        weights[~d_mask] = 0

    geo_loss_value = torch.tensor(0.0, device=device)
    if geo_weight > 0 and d_mask.sum() > 0:
        geo_loss_value = geo_weight * ((weights * pixel_noise)[d_mask]).mean()

    return {
        "d_mask": d_mask,                       # (HW,) bool
        "map_z": map_z,
        "weights": weights,                     # (HW,) float
        "pixel_noise": pixel_noise,             # (HW,) float
        "pts": pts,                             # (HW, 3)
        "geo_loss": geo_loss_value,             # float
        "device": device,
        "pixels": pixels
    }


def multi_view_normal_consistency(
    nearest_normal, nearest_normal_surf,
    ref_normal, ref_normal_surf,
    nearest_cam,
    pts,
    map_z,
    d_mask,
    valid_indices,
    sample_num_normal,
    normal_consistency_weight=0.05,
    pixel_noise_th=1.0,

):
    """
    Untested multi-view normal loss, not used in the paper
    """

    device = ref_normal.device

    #Rt_nearest = torch.tensor(nearest_cam.world_view_transform, dtype=torch.float32, device=device)  # (4,4)
    Rt_nearest = nearest_cam.world_view_transform.clone().detach().to(dtype=torch.float32, device=device)

    # Nx3 => Nx4
    N = pts.shape[0]
    ones = torch.ones((N, 1), device=device, dtype=torch.float32)
    pts_world_homo = torch.cat([pts, ones], dim=-1)
    pts_in_nearest_homo = pts_world_homo @ Rt_nearest.T
    pts_in_nearest = pts_in_nearest_homo[:, :3]

    px_nearest = (pts_in_nearest[:, 0] / (pts_in_nearest[:, 2] + 1e-8)) * nearest_cam.Fx + nearest_cam.Cx
    py_nearest = (pts_in_nearest[:, 1] / (pts_in_nearest[:, 2] + 1e-8)) * nearest_cam.Fy + nearest_cam.Cy

    px_sub = px_nearest[valid_indices]
    py_sub = py_nearest[valid_indices]

    Hn, Wn = nearest_cam.image_height, nearest_cam.image_width
    in_image_mask = (px_sub >= 0) & (px_sub < Wn - 1) & (py_sub >= 0) & (py_sub < Hn - 1)

    if in_image_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    valid_indices_sub = valid_indices[in_image_mask]
    px_subset = px_sub[in_image_mask]
    py_subset = py_sub[in_image_mask]

    uv_x = 2.0 * px_subset / (Wn - 1) - 1.0
    uv_y = 2.0 * py_subset / (Hn - 1) - 1.0
    uv = torch.stack([uv_x, uv_y], dim=-1).view(1, -1, 1, 2)

    nearest_n_b = nearest_normal.unsqueeze(0)
    nearest_n_surf_b = nearest_normal_surf.unsqueeze(0)

    sampled_nearest_n = F.grid_sample(nearest_n_b, uv, align_corners=True)
    sampled_nearest_n = sampled_nearest_n.squeeze(0).squeeze(-1).permute(1, 0)  # (N_valid, 3)

    sampled_nearest_n_surf = F.grid_sample(nearest_n_surf_b, uv, align_corners=True)
    sampled_nearest_n_surf = sampled_nearest_n_surf.squeeze(0).squeeze(-1).permute(1, 0)  # (N_valid, 3)

    ref_n_ = ref_normal.permute(1, 2, 0).reshape(-1, 3)
    ref_n_surf_ = ref_normal_surf.permute(1, 2, 0).reshape(-1, 3)

    valid_ref_n = ref_n_[valid_indices_sub]           # (N_valid, 3)
    valid_ref_n_surf = ref_n_surf_[valid_indices_sub] # (N_valid, 3)

    ref_norm = F.normalize(valid_ref_n, dim=-1)
    ref_norm_surf = F.normalize(valid_ref_n_surf, dim=-1)

    near_norm = F.normalize(sampled_nearest_n, dim=-1)
    near_norm_surf = F.normalize(sampled_nearest_n_surf, dim=-1)

    dot_prod = torch.abs(ref_norm * near_norm).sum(dim=-1)
    dot_prod_surf = torch.abs(ref_norm_surf * near_norm_surf).sum(dim=-1)

    normal_diff = 1.0 - dot_prod
    normal_diff_surf = 1.0 - dot_prod_surf

    normal_mv_loss = normal_consistency_weight * (normal_diff.mean() + normal_diff_surf.mean())
    return normal_mv_loss