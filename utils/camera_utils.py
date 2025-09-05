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
import cv2

from scene.cameras import Camera
import numpy as np

from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import sys

from PIL import Image
import os
import torch.nn.functional as F

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, test):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    if not test:
        if args.use_normal_prior == 1:
            import torch
            # normal_path = cam_info.image_path.replace('images_4', args.w_normal_prior)
            normal_path = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), args.w_normal_prior,
                                       os.path.basename(cam_info.image_path))

            if os.path.exists(normal_path[:-4] + '.npy'):
                _normal = torch.tensor(np.load(normal_path[:-4] + '.npy'))
                _normal = - (_normal * 2 - 1)
                resized_normal = F.interpolate(_normal.unsqueeze(0), size=resolution[::-1], mode='bicubic')
                _normal = resized_normal.squeeze(0)
            else:
                _normal = Image.open(normal_path[:-4] + '.png')
                resized_normal = PILtoTorch(_normal, resolution)
                resized_normal = resized_normal[:3]
                _normal = - (resized_normal * 2 - 1)

            # normalize normal
            _normal = _normal.permute(1, 2, 0) @ (torch.tensor(np.linalg.inv(cam_info.R)).float())
            _normal = _normal.permute(2, 0, 1)

        else:
            _normal = None

        if args.use_depth_prior == 1:
            import torch
            depth_path = os.path.join(os.path.dirname(os.path.dirname(cam_info.image_path)), "depth_pro",
                                      args.w_depth_prior)
            depth_path = depth_path + "/" + cam_info.image_name

            if os.path.exists(depth_path + '.npy'):
                _depth = torch.tensor(np.load(depth_path + '.npy').transpose(1, 0)).unsqueeze(0).unsqueeze(0)
                resized_depth = F.interpolate(_depth, size=resolution, mode='bicubic')
                _depth = resized_depth.squeeze(0)

            else:
                _depth = cv2.imread(depth_path + '.png', -1).astype(np.float32) / float(2 ** 16)
                _depth = cv2.resize(_depth, resolution)
                # _depth = Image.open(depth_path + '.png')
                # resized_depth = PILtoTorch(_depth, resolution)
                # resized_depth = resized_depth[:3]
                # _depth = - (resized_depth * 2 - 1)
            # normalize depth
            _depth = _depth.permute(0, 2, 1)
        else:
            _depth = None
    else:
        _normal = None
        _depth = None


    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=cam_info.global_id,
                  preload_img=args.preload_img,
                  normal=_normal, depth_geo=_depth,
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device, white_background=args.white_background)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, test):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, test=test))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
