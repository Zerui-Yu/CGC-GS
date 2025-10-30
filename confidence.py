import cv2
import torch
import os
import matplotlib.pyplot as plt
from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.point_utils import depth_to_normal
import numpy as np

def linear_schedule(current_round, start_value, end_value, start_round=7000, end_round=15000):

    if current_round <= start_round:
        return start_value
    elif current_round >= end_round:
        return end_value
    else:
        ratio = (current_round - start_round) / (end_round - start_round)
        return start_value + ratio * (end_value - start_value)

def compute_angular_error(normal1, normal2,use_abs=True):
    dot = (normal1 * normal2).sum(dim=0)  # (H, W)
    if use_abs:
        dot = torch.abs(dot)  # (H, W)
    dot = torch.clamp(dot, -1.0, 1.0)
    error = torch.acos(dot)

    return error


def compute_confidence_from_angular_error(error, sigma=0.2):
    confidence = torch.exp(- (error ** 2) / (2 * sigma ** 2))

    return confidence


def render_set(views, gaussians, pipeline, background, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _, _ = view.get_image()
        out = render(view, gaussians, pipeline, background)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        pgsr_dpnormal = out["depth_normal"]
        normal_mono = view.normal.to("cuda")

        angular_error = compute_angular_error(pgsr_dpnormal, normal_mono)
        confidence = compute_confidence_from_angular_error(angular_error, sigma=0.2)

        angular_error_np = angular_error.cpu().numpy()
        confidence_np = confidence.cpu().numpy()

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(angular_error_np, cmap='jet')
        plt.title("Angular Error (rad)")
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(confidence_np, cmap='hot')
        plt.title("Confidence Map")
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"view_{idx:03d}_error_confidence.png")
        plt.savefig(save_path)
        plt.close()


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, save_dir: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(scene.getTrainCameras(), gaussians, pipeline, background, save_dir)


# if needed, run confidence.py to render angular error and confidence map
if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    save_dir = args.model_path + "/vis_abs"
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), save_dir)
