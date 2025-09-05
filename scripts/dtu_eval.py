import os
from argparse import ArgumentParser
import open3d as o3d
import numpy as np
import torch
from PIL import Image

from utils.general_utils import PILtoTorch


dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument('--dtu', "-dtu", required=True, type=str)
args, _ = parser.parse_known_args()

skip_training = 0
skip_rendering = 0
skip_mesh = 0
skip_metrics = 0

all_scenes = []
all_scenes.extend(dtu_scenes)

script_dir = os.path.dirname(os.path.abspath(__file__))

output_path = "./dtu_test/"

for scene in dtu_scenes:
    source = args.dtu + "/" + scene
    common_args_train = " --quiet -r 2 --test_iterations -1 --use_depth_prior 1 --use_normal_prior 1"
    common_args_rendering = " --quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    if skip_training == 0:
        print(
            "python train.py -s " + source + " -m " + output_path + "/" + scene + common_args_train + " --data_device cuda")
        os.system(
            "python train.py -s " + source + " -m " + output_path + "/" + scene + common_args_train + " --data_device cuda")

    if skip_rendering == 0:
        print(
            f"python render.py" + " -m" + output_path + "/" + scene + common_args_rendering)
        os.system(
            f"python render.py" + " -m" + output_path + "/" + scene + common_args_rendering)

    if skip_mesh == 0:
        scan_id = scene[4:]
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
                 f"--input_mesh {output_path}/{scene}/mesh/tsdf_fusion_post.ply " + \
                 f"--scan_id {scan_id} --output_dir ./eval/{path}/scan{scan_id} " + \
                 f"--mask_dir ./data/DTU " + \
                 f"--DTU ./data/DTU_points"

        os.system(string)

    if skip_metrics == 0:
        print(
            f"python metrics.py" + " -m" + output_path + "/" + scene)
        os.system(
            f"python metrics.py" + " -m" + output_path + "/" + scene + " -t train")
os.system(f"python ./scripts/summary.py -m" + output_path)

