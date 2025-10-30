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

import os
import torch
import random
import numpy as np

from multiview_loss import compute_geometry_mask_and_weight
from scene.app_model import AppModel
from scene.cameras import Camera
from random import randint

from scene.cameras import normalize_depth
from utils.graphics_utils import patch_offsets, patch_warp
from utils.loss_utils import l1_loss, ssim, lncc, scale_invariant_gradient_loss_ori, cof_guide_normal_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def culling(xyz, cams, expansion=2):
    cam_centers = torch.stack([c.camera_center for c in cams], 0).to(xyz.device)
    span_x = cam_centers[:, 0].max() - cam_centers[:, 0].min()
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() # smallest span
    span_z = cam_centers[:, 2].max() - cam_centers[:, 2].min()

    scene_center = cam_centers.mean(0)

    span_x = span_x * expansion
    span_y = span_y * expansion
    span_z = span_z * expansion

    x_min = scene_center[0] - span_x / 2
    x_max = scene_center[0] + span_x / 2

    y_min = scene_center[1] - span_y / 2
    y_max = scene_center[1] + span_y / 2

    z_min = scene_center[2] - span_x / 2
    z_max = scene_center[2] + span_x / 2


    valid_mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
                 (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
                 (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    # print(f'scene mask ratio {valid_mask.sum().item() / valid_mask.shape[0]}')

    return valid_mask, scene_center


def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                        preload_img=False, data_device="cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_imgloss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    #ema_gamma_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_multi_view_cen_for_log = 0.0
    #ema_multi_view_norm_for_log = 0.0
    ema_mask_for_log = 0.0

    image_loss, dist_loss, normal_loss, depth_loss, lambda_depth, geo_loss, ncc_loss, census_loss, normal_mv_loss, mask_loss = None, None, None, None, None, None, None, None, None, None
    threshold_gamma = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, app_model=app_model, depth_threshold=opt.depth_threshold)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image, gt_image_gray, gt_image_census = viewpoint_cam.get_image()

        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        gt_alpha = (
            viewpoint_cam.mask.cuda()
            if viewpoint_cam.mask is not None
            else None
        )

        if gt_alpha is not None and opt.use_mask == 1:
            alpha = render_pkg["rend_alpha"]
            background_mask = (gt_alpha == 0)
            background_mask_alpha = (gt_alpha == 0)
            if background_mask.dim() == 2:
                background_mask = background_mask.unsqueeze(0)

            if background_mask.shape[0] != image.shape[0]:
                background_mask = background_mask.expand(image.shape[0], -1, -1)

            if args.white_background:
                target_value = torch.ones_like(image)
            else:
                target_value = torch.zeros_like(image)
            
            lambda_mask = opt.lambda_mask

            mask_loss = (lambda_mask * torch.nn.functional.l1_loss(image * background_mask, target_value * background_mask) +
                         torch.nn.functional.l1_loss(alpha*background_mask_alpha, gt_alpha*background_mask_alpha))

        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        if mask_loss is not None and iteration > 3000:
            loss += mask_loss

        # regularization
        if iteration > 7000 and opt.use_norm_loss == 1:

            rend_normal = render_pkg['normal']
            surf_normal = render_pkg['depth_normal']

            if viewpoint_cam.normal is not None:
                prior_normal = viewpoint_cam.normal.to("cuda").detach()
            else:
                prior_normal = None

            normal_loss = cof_guide_normal_loss(rend_normal, surf_normal, prior_normal, gt_image, opt.use_prior_n, opt.use_cof, opt.cof_gamma, opt.lambda_normal)

            loss += normal_loss

        if iteration > 3000 and opt.use_prior_d == 1:
            lambda_depth = depth_l1_weight(iteration)
            depth = render_pkg['depth']
            if viewpoint_cam.depth_geo is not None:

                #TODO: Add support for invdepth
                #inv_depth = render_pkg['invdepth']

                prior_depth = viewpoint_cam.depth_geo.to("cuda").detach()
                Ll1depth_pure = scale_invariant_gradient_loss_ori(depth, prior_depth, mask=viewpoint_cam.mask, use_mask=opt.use_mask)
                Ll1depth = lambda_depth * Ll1depth_pure
                depth_loss = Ll1depth
                loss += depth_loss

        if iteration > 3000:
            lambda_dist = opt.lambda_dist
            rend_dist = render_pkg["rend_dist"]
            dist_loss = lambda_dist * (rend_dist).mean()
            loss += dist_loss

        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter and opt.use_multi_loss == 1:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id, 1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < 0.5 or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                ## compute geometry consistency mask and loss

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, depth_threshold=opt.depth_threshold, app_model=app_model)

                geo_info = compute_geometry_mask_and_weight(
                    ref_cam=viewpoint_cam,
                    nearest_cam=nearest_cam,
                    ref_depth=render_pkg["depth"],
                    nearest_depth=nearest_render_pkg["depth"],
                    gaussians=gaussians,
                    pixel_noise_th=opt.multi_view_pixel_noise_th,
                    wo_use_geo_occ_aware=opt.wo_use_geo_occ_aware,
                    geo_weight=opt.multi_view_geo_weight
                )

                d_mask = geo_info["d_mask"]  # shape (H*W,)
                weights = geo_info["weights"]  # shape (H*W,)
                pixels_grid = geo_info["pixels"]
                geo_loss = geo_info["geo_loss"]

                if opt.use_geo == 1:
                    loss += geo_loss

                if d_mask.sum() > 0:
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels_grid.reshape(-1, 2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                                         align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_census_ = gt_image_census.unsqueeze(0)
                            ref_patch_census = F.grid_sample(ref_census_.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_patch_census = ref_patch_census.view(-1, total_patch_size)

                            #TODO: If the batch of ref_census != 0
                            #ref_patch_census =...

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) @ viewpoint_cam.world_view_transform[:3, :3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

                        ## compute Homography
                        ref_local_n = render_pkg["normal"].permute(1, 2, 0)
                        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                        ref_local_d = render_pkg['distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                                            torch.matmul(
                                                ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                                ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2,
                                                                                                                   1)) / \
                                            ref_local_d[..., None, None]
                        H_ref_to_neareast = torch.matmul(
                            nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
                            H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray, nearest_image_census = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                        near_census_ = nearest_image_census.unsqueeze(0)
                        near_patch_census = F.grid_sample(near_census_.unsqueeze(1), grid.reshape(1, -1, 1, 2), align_corners=True)
                        near_patch_census = near_patch_census.view(-1, total_patch_size)

                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0 and opt.use_ncc == 1:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

                        census_diff = (ref_patch_census - near_patch_census) ** 2
                        census_diff_norm = torch.mean(census_diff / (0.1 + census_diff), dim=1)
                        census_diff_norm = census_diff_norm * weights

                        census_diff_filtered = census_diff_norm[mask]

                        if census_diff_norm.numel() > 0 and opt.use_census == 1:
                            # => scalar
                            census_loss = opt.lambda_census * census_diff_filtered.mean()
                            loss += census_loss

                        #TODO: Patch-based normal loss
                        #normal_mv_loss = multi_view_normal_consistency(nearest_normal=render_pkg["normal"],nearest_normal_surf=render_pkg["depth_normal"],
                                                                       #ref_normal=nearest_render_pkg["normal"],ref_normal_surf=nearest_render_pkg["depth_normal"],
                                                                       #nearest_cam=nearest_cam,
                                                                       #valid_indices=valid_indices,
                                                                       #pts=geo_info["pts"],
                                                                       #map_z=geo_info["map_z"],
                                                                       #d_mask=geo_info["d_mask"],
                                                                       #sample_num_normal=opt.multi_view_sample_num,
                                                                       #normal_consistency_weight=opt.normal_consistency_weight)
                        #loss += normal_mv_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_imgloss_for_log = 0.4 * image_loss.item() if image_loss is not None else 0.0 + 0.6 * ema_imgloss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() if dist_loss is not None else 0.0 + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_normal_for_log
            #ema_gamma_for_log = threshold_gamma if threshold_gamma is not None else opt.cof_gamma
            ema_depth_for_log = 0.4 * depth_loss.item() if depth_loss is not None else 0.0 + 0.6 * ema_depth_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            ema_multi_view_cen_for_log = 0.4 * census_loss.item() if census_loss is not None else 0.0 + 0.6 * ema_multi_view_cen_for_log
            #ema_multi_view_norm_for_log = 0.4 * normal_mv_loss.item() if normal_mv_loss is not None else 0.0 + 0.6 * ema_multi_view_norm_for_log
            ema_mask_for_log = 0.4 * mask_loss.item() if mask_loss is not None else 0.0 + 0.6 * ema_mask_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "ImageLoss": f"{ema_imgloss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    #"γ": f"{ema_gamma_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Cen": f"{ema_multi_view_cen_for_log:.{5}f}",
                    #"mv_norm": f"{ema_multi_view_norm_for_log:.{5}f}",
                    "mask": f"{ema_mask_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model=app_model,)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            #Densify Control
            if iteration < 3000:
                split = "ordinary"
            elif iteration < (3000+opt.scale_iter1):
                split = opt.split
            elif iteration < 9000:
                split = "ordinary"
            elif iteration < (9000+opt.scale_iter2):
                split = opt.split
            elif iteration < opt.densify_until_iter:
                split = "ordinary"

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, render_pkg["pixels"])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    if split == "ordinary":
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, max_points=opt.max_points)

                    elif split == "scale_0":
                        scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                        gaussians.densify_and_scale_split(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, opt.max_screen_size, opt.percent_dense, opt.scale_factor, scene_mask, N=2, no_grad=False, max_points=opt.max_points)

                    elif split == "scale_1":
                        scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                        gaussians.densify_and_scale_split(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, opt.max_screen_size, opt.percent_dense, opt.scale_factor, scene_mask, N=2, no_grad=True, max_points=opt.max_points)

                    elif split == "none":
                        pass
                    else:
                        raise ValueError(f"Unknown split type {split}")

            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                opacity_reset_interval = opt.opacity_reset_interval
                if iteration > 5000:
                    opacity_reset_interval = opt.opacity_reset_interval
                if iteration % opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # opacity decay
            #if iteration < opt.densify_until_iter:
            #    if iteration % opt.opacity_decay_interval == 0:
            #        gaussians.scale_all_opacity(factor=opt.opacity_decay_factor)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                app_model.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer, depth_threshold=opt.depth_threshold)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()



def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    #render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg['render']
                    if 'app_image' in render_pkg:
                        image = render_pkg['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_normal = render_pkg["normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["depth_normal"] * 0.5 + 0.5

                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)


                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),gt_image[None], global_step=iteration)
                            if viewpoint.normal is not None:
                                prior_normal = viewpoint.normal * 0.5 + 0.5
                                tb_writer.add_images(config['name'] + "_view_{}/prior_normal".format(viewpoint.image_name),prior_normal[None], global_step=iteration)


                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
