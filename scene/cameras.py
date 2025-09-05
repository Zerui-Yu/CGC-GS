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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrixCenterShift
import copy
from PIL import Image
from utils.general_utils import PILtoTorch
import os, cv2
import torch.nn.functional as F


def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def normalize_depth(depth):
    # depth: [1, H, W]
    min_val = torch.min(depth)
    max_val = torch.max(depth)
    normalized = (depth - min_val) / (max_val - min_val)
    return normalized

def process_image(image_path, resolution, ncc_scale, white_background=False):

    if 'nerf_synthetic' in os.path.normpath(image_path).split(os.sep):
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        alpha = norm_data[:, :, 3:4]
        rgba = np.concatenate([arr, alpha], axis=2)
        image = Image.fromarray(np.array(rgba * 255.0, dtype=np.uint8), "RGBA")
    else:
        image = Image.open(image_path)

    if len(image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0] / ncc_scale), int(resolution[1] / ncc_scale))
            resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0] / ncc_scale), int(resolution[1] / ncc_scale))
            resized_image_rgb = PILtoTorch(image, ncc_resolution)
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]

    # Compute Census Feature Map
    census_gray = compute_ternary_transform_gpu(gray_image, kernel_size=3)

    return gt_image, gray_image, loaded_mask, census_gray


def compute_ternary_transform_gpu(
    gray_image: torch.Tensor,
    kernel_size: int = 3,
    eps: float = 0.81
) -> torch.Tensor:
    """
    Compute 'Ternary transform' in GPU.
    """

    assert gray_image.ndim == 3 and gray_image.shape[0] == 1, \
        "gray_image should be [1,H,W]"

    device = gray_image.device
    _, H, W = gray_image.shape

    pad = (kernel_size - 1) // 2
    patches = F.unfold(
        gray_image.unsqueeze(0),
        kernel_size=kernel_size,
        padding=pad,
        stride=1
    )

    patches = patches.squeeze(0)
    patches = patches.permute(1,0)

    # skip center pixel
    center_idx = (kernel_size * kernel_size) // 2
    idxs = list(range(kernel_size*kernel_size))
    idxs.remove(center_idx)
    neighbors = patches[:, idxs]
    center = patches[:, center_idx]

    diff = neighbors - center.unsqueeze(1)
    denom = torch.sqrt(diff*diff + eps)
    ternary = diff / denom

    # reshape
    ternary = ternary.permute(1,0)
    ternary_map = ternary.view((kernel_size*kernel_size -1), H, W)
    ternary_map = torch.mean(ternary_map,dim=0)

    return ternary_map.squeeze()

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 image_width, image_height,
                 image_path, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 ncc_scale=1.0,
                 preload_img=True, data_device="cuda",normal=None, depth_geo=None, white_background=False
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        self.white_background = white_background
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image, self.image_gray, self.mask, self.gray_census = None, None, None, None

        # Prior
        if normal is not None:
            #self.normal = normal.to(self.data_device)
            self.normal = normal.to("cpu")
            normal_norm = torch.norm(self.normal, dim=0, keepdim=True)
            self.normal_mask = ~((normal_norm > 1.1) | (normal_norm < 0.9))
            self.normal = self.normal / normal_norm
        else:
            self.normal = None
            self.normal_mask = None

        if depth_geo is not None:
            #self.depth_geo = depth_geo.to(self.data_device)
            self.depth_geo = depth_geo.to("cpu")
            self.depth_geo_mask = None

        else:
            self.depth_geo = None
            self.depth_geo_mask = None

        self.preload_img = preload_img
        self.ncc_scale = ncc_scale
        if self.preload_img:
            gt_image, gray_image, loaded_mask, gray_census = process_image(self.image_path, self.resolution, ncc_scale, self.white_background)
            self.original_image = gt_image.to(self.data_device)
            self.original_image_gray = gray_image.to(self.data_device)
            self.mask = loaded_mask
            self.gray_census = gray_census.to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.plane_mask, self.non_plane_mask = None, None

    def get_image(self):
        if self.preload_img:
            return self.original_image.cuda(), self.original_image_gray.cuda(), self.gray_census.cuda()
        else:
            gt_image, gray_image, _ , gray_census = process_image(self.image_path, self.resolution, self.ncc_scale, self.white_background)
            return gt_image.cuda(), gray_image.cuda(), gray_census.cuda()

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor(
            [[self.Fx / scale, 0, self.Cx / scale], [0, self.Fy / scale, self.Cy / scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous()  # cam2world
        return intrinsic_matrix, extrinsic_matrix

    def get_rays(self, scale=1.0):
        W, H = int(self.image_width / scale), int(self.image_height / scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
            [(ix - self.Cx / scale) / self.Fx * scale,
             (iy - self.Cy / scale) / self.Fy * scale,
             torch.ones_like(ix)], -1).float().cuda()
        return rays_d

    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                          [0, self.Fy / scale, self.Cy / scale],
                          [0, 0, 1]]).cuda()
        return K

    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale / self.Fx, 0, -self.Cx / self.Fx],
                            [0, scale / self.Fy, -self.Cy / self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def sample_cam(cam_l: Camera, cam_r: Camera):
    cam = copy.copy(cam_l)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam_l.R.transpose()
    Rt[:3, 3] = cam_l.T
    Rt[3, 3] = 1.0

    Rt2 = np.zeros((4, 4))
    Rt2[:3, :3] = cam_r.R.transpose()
    Rt2[:3, 3] = cam_r.T
    Rt2[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    C2W2 = np.linalg.inv(Rt2)
    w = np.random.rand()
    pose_c2w_at_unseen = w * C2W + (1 - w) * C2W2
    Rt = np.linalg.inv(pose_c2w_at_unseen)
    cam.R = Rt[:3, :3]
    cam.T = Rt[:3, 3]

    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(
        0, 1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam