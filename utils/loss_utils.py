#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

from confidence import compute_angular_error, compute_confidence_from_angular_error


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_loss_mask(network_output, gt, mask):
    pixels = torch.sum(mask)
    return torch.sum(torch.abs((network_output - gt))) / pixels


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:, 1:-1, :-2] + disp[:, 1:-1, 2:] - 2 * disp[:, 1:-1, 1:-1])
    grad_disp_y = torch.abs(disp[:, :-2, 1:-1] + disp[:, 2:, 1:-1] - 2 * disp[:, 1:-1, 1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def edge_aware_curvature_loss(I, D, mask=None):
    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device) / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(I.device) / 4

    # Compute derivatives of D
    dD_dx = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_x, padding=1) for i in range(D.shape[0])])
    dD_dy = torch.cat([F.conv2d(D[i].unsqueeze(0), sobel_y, padding=1) for i in range(D.shape[0])])

    # Compute derivatives of I
    dI_dx = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_x, padding=1) for i in range(I.shape[0])])
    dI_dx = torch.mean(torch.abs(dI_dx), 0, keepdim=True)
    dI_dy = torch.cat([F.conv2d(I[i].unsqueeze(0), sobel_y, padding=1) for i in range(I.shape[0])])
    dI_dy = torch.mean(torch.abs(dI_dy), 0, keepdim=True)

    # Compute weights
    weights_x = (dI_dx - 1) ** 500
    weights_y = (dI_dy - 1) ** 500

    # Compute losses
    loss_x = torch.abs(dD_dx) * weights_x
    loss_y = torch.abs(dD_dy) * weights_y

    # Apply mask to losses
    if mask is not None:
        # Ensure mask is on the correct device and has correct dimensions
        mask = mask.to(I.device)
        loss_x = loss_x * mask
        loss_y = loss_y * mask

        # Count valid pixels
        valid_pixel_count = mask.sum()

        # Compute the mean loss only over valid pixels
        if valid_pixel_count.item() > 0:
            return (loss_x.sum() + loss_y.sum()) / valid_pixel_count
        else:
            # Handle the case where no valid pixels exist
            return torch.tensor(0.0, device=I.device, requires_grad=True)
    else:
        # If no mask is provided, calculate the mean over all pixels
        return (loss_x + loss_y).mean()


def scale_invariant_gradient_loss_ori(pred_depth, gt_depth, eps=1e-6, mask=None, use_mask=0):
    """
    our scale invariant depth loss
    Parm：
        pred_depth: [1, H, W]
        gt_depth: [1, H, W]
    """

    log_pred = torch.log(pred_depth + eps)
    log_gt = torch.log(gt_depth + eps)

    if use_mask == 1:
        mask = mask.to("cuda")
        log_pred = log_pred * mask
        log_gt = log_gt * mask

    grad_pred_x = log_pred[:, :, 1:] - log_pred[:, :, :-1]
    grad_gt_x = log_gt[:, :, 1:] - log_gt[:, :, :-1]

    grad_pred_y = log_pred[:, 1:, :] - log_pred[:, :-1, :]
    grad_gt_y = log_gt[:, 1:, :] - log_gt[:, :-1, :]

    grad_pred_x = torch.nan_to_num(grad_pred_x, nan=0.0)
    grad_pred_y = torch.nan_to_num(grad_pred_y, nan=0.0)
    grad_gt_x = torch.nan_to_num(grad_gt_x, nan=0.0)
    grad_gt_y = torch.nan_to_num(grad_gt_y, nan=0.0)

    loss_x = torch.abs(grad_pred_x - grad_gt_x).mean()
    loss_y = torch.abs(grad_pred_y - grad_gt_y).mean()

    return loss_x + loss_y


def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape
    bottom_point = img[..., 2:hd, 1:wd - 1]
    top_point = img[..., 0:hd - 2, 1:wd - 1]
    right_point = img[..., 1:hd - 1, 2:wd]
    left_point = img[..., 1:hd - 1, 0:wd - 2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None, None], (1, 1, 1, 1), mode='constant', value=1.0).squeeze()
    return grad_img


def cof_guide_normal_loss(rend_normal, surf_normal, prior_normal, gt_image, use_prior_n=1, use_cof=1, cof_gamma=0.2,
                          lambda_normal=0.02):
    if prior_normal is not None and use_prior_n == 1 and use_cof == 1:

        angular_error = compute_angular_error(surf_normal, prior_normal, use_abs=True)
        confidence = compute_confidence_from_angular_error(angular_error, sigma=0.2)

        # gamma in the Eq.(11)
        # threshold_gamma = linear_schedule(iteration, start_value=opt.cof_gamma, end_value=0.5)
        mask_high_conf = (confidence >= cof_gamma)

        # —— Prepare Normal Loss——
        # 1) cof >= gamma
        normal_error_prior_0 = 1 - torch.abs((rend_normal * prior_normal).sum(dim=0))
        normal_error_prior_1 = 1 - torch.abs((surf_normal * prior_normal).sum(dim=0))
        # normal_error_prior_0 = 1 - (rend_normal * prior_normal).sum(dim=0)
        # normal_error_prior_1 = 1 - (surf_normal * prior_normal).sum(dim=0)

        # 2) cof < gamma
        normal_error_surf = 1 - torch.abs((rend_normal * surf_normal).sum(dim=0))  # (H, W)
        # normal_error_surf = 1 - (rend_normal * surf_normal).sum(dim=0) # (H, W)

        # Eq.(11)
        final_normal_error = torch.where(mask_high_conf, (normal_error_prior_0 + normal_error_prior_1),
                                         2 * normal_error_surf)

        # with the gradient of gt
        final_normal_error = final_normal_error[None]
        image_weight = (1.0 - get_img_grad_weight(gt_image))
        image_weight = ((image_weight).clamp(0, 1).detach() ** 2).unsqueeze(0)
        final_normal_error = image_weight * final_normal_error
        normal_loss = lambda_normal * final_normal_error.mean()

    elif prior_normal is not None and use_prior_n == 1 and use_cof == 0:

        # normal_error_prior_0 = 1 - torch.abs((rend_normal * prior_normal).sum(dim=0))  # (H, W)
        # normal_error_prior_1 = 1 - torch.abs((surf_normal * prior_normal).sum(dim=0))
        normal_error_prior_0 = 1 - (rend_normal * prior_normal).sum(dim=0)  # (H, W)
        normal_error_prior_1 = 1 - (surf_normal * prior_normal).sum(dim=0)

        final_normal_error = normal_error_prior_0 + normal_error_prior_1
        final_normal_error = final_normal_error[None]
        image_weight = (1.0 - get_img_grad_weight(gt_image))
        image_weight = ((image_weight).clamp(0, 1).detach() ** 2).unsqueeze(0)
        final_normal_error = image_weight * final_normal_error
        normal_loss = lambda_normal * final_normal_error.mean()

    else:
        normal_error = (1 - torch.abs((rend_normal * surf_normal).sum(dim=0)))[None]
        # normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        image_weight = (1.0 - get_img_grad_weight(gt_image))
        image_weight = ((image_weight).clamp(0, 1).detach() ** 2).unsqueeze(0)
        normal_error = image_weight * normal_error
        normal_loss = lambda_normal * (normal_error).mean()

    return normal_loss


def scale_invariant_gradient_loss(
        pred_depth, gt_depth, eps=1e-6, mask=None, use_mask=0,
        p_low=0.10, p_high=0.90
):
    with torch.no_grad():
        dev = gt_depth.device
        valid = torch.isfinite(gt_depth) & (gt_depth > eps)

        if valid.any():
            ql, qh = torch.quantile(
                gt_depth[valid].float(), torch.tensor([p_low, p_high], device=dev)
            )
            valid = valid & (gt_depth >= ql) & (gt_depth <= qh)

        if (mask is not None) and (use_mask == 1):
            valid = valid & (mask.to(dev).bool())

    log_pred = torch.log(pred_depth + eps)
    log_gt = torch.log(gt_depth + eps)

    grad_pred_x = log_pred[:, :, 1:] - log_pred[:, :, :-1]
    grad_gt_x = log_gt[:, :, 1:] - log_gt[:, :, :-1]
    grad_pred_y = log_pred[:, 1:, :] - log_pred[:, :-1, :]
    grad_gt_y = log_gt[:, 1:, :] - log_gt[:, :-1, :]

    valid_x = valid[:, :, 1:] & valid[:, :, :-1]
    valid_y = valid[:, 1:, :] & valid[:, :-1, :]

    grad_pred_x = torch.nan_to_num(grad_pred_x, nan=0.0, posinf=0.0, neginf=0.0)
    grad_pred_y = torch.nan_to_num(grad_pred_y, nan=0.0, posinf=0.0, neginf=0.0)
    grad_gt_x = torch.nan_to_num(grad_gt_x, nan=0.0, posinf=0.0, neginf=0.0)
    grad_gt_y = torch.nan_to_num(grad_gt_y, nan=0.0, posinf=0.0, neginf=0.0)

    diff_x = (grad_pred_x - grad_gt_x).abs()
    diff_y = (grad_pred_y - grad_gt_y).abs()

    if valid_x.any():
        loss_x = diff_x[valid_x].mean()
    else:
        loss_x = torch.tensor(0.0, device=gt_depth.device)

    if valid_y.any():
        loss_y = diff_y[valid_y].mean()
    else:
        loss_y = torch.tensor(0.0, device=gt_depth.device)

    return loss_x + loss_y


def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask
