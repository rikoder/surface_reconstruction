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
import copy
import json
import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import torchvision.transforms.functional as tf

import numpy as np
import matplotlib.cm as cm
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path

import matplotlib.pyplot as plt
from PIL import Image

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def vis_depth(depth):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    percentile = 99
    #eps = 1e-10
    eps = np.finfo(np.float32).eps

    lo_auto, hi_auto = weighted_percentile(
        depth, np.ones_like(depth), [50 - percentile / 2, 50 + percentile / 2])
    lo = None or (lo_auto - eps)
    hi = None or (hi_auto + eps)
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps) #lambda x: 1/x + eps

    depth, lo, hi = [curve_fn(x) for x in [depth, lo, hi]]
    depth = np.nan_to_num(
            np.clip((depth - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))
    colorized = cm.get_cmap('turbo')(depth)[:, :, :3]

    return np.uint8(colorized[..., ::-1] * 255)

def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

    Returns:
    A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return colorized

depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

depth_diff_curve = lambda x: x #np.log(x + 1e-5) 


def get_depth_img(ground_truth_depth_path):
   
    # Load ground truth depth image
    ground_truth_depth = Image.open(ground_truth_depth_path)
    ground_truth_depth = tf.to_tensor(ground_truth_depth).unsqueeze(0)[:, :3, :, :].cuda()
    
    # ground_truth_depth = torch.from_numpy(np.array(ground_truth_depth)).float()  # Convert to tensor
    
    # ground_truth_depth = ground_truth_depth.to('cuda')

    # if ground_truth_depth.dim() == 3 and ground_truth_depth.shape[2] == 3:
    #     ground_truth_depth = ground_truth_depth.mean(dim=2, keepdim=True)  # Average across channels
    #     #print("Converted Ground Truth Depth Shape:", ground_truth_depth.shape)  # Should be [756, 1008, 1]

    # # Ensure ground_truth_depth is [1, 756, 1008]
    # ground_truth_depth = ground_truth_depth.permute(2, 0, 1)

    # #print(ground_truth_depth.shape,depth.shape)

    # # Normalize ground truth depth to match the range of predicted depth
    # gt_min, gt_max = ground_truth_depth.min(), ground_truth_depth.max()

    # # Normalizing ground_truth_depth
    # normalized_ground_truth_depth = (ground_truth_depth - gt_min) / (gt_max - gt_min)

    return ground_truth_depth

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, depth_GT, GT_iters):
    near=0
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    if near > 0:
        mask_near = None
        for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
            mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near
            mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
        gaussians.prune_points_inference(mask_near)


    pbar = tqdm(views, desc="Rendering progress")

    # Saving the scales 
    # print(dataset.model_path)
    scene_name = os.path.basename(model_path)
    scaling_copy = gaussians.get_scaling
    file_path = os.path.join('/home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/scales', f"scaling_copy_{scene_name}.pt")
    torch.save(scaling_copy, file_path)

    # Assuming depth_GT, model_path, and GT_iters are defined and available
    for idx, view in enumerate(pbar):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        
        # Ground truth RGB image
        gt = view.original_image[0:3, :, :]

        # Predicted depth calculation
        depth = (render_pkg['depth'] - render_pkg['depth'].min()) / (render_pkg['depth'].max() - render_pkg['depth'].min()) + (1 * (1 - render_pkg["alpha"]))

        # Save rendered images
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}.png'.format(idx)))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}.png'.format(idx)))
        torchvision.utils.save_image(1 - depth, os.path.join(depth_path, '{0:05d}.png'.format(idx)))
        torchvision.utils.save_image(render_pkg["alpha"], os.path.join(depth_path, 'alpha_{0:05d}.png'.format(idx)))

        # Visualize depth map (optional, as in your original code)
        #print(depth.shape)
        depth_est = depth.squeeze().cpu().numpy()
        depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
        depth_est = torch.as_tensor(depth_est).permute(2, 0, 1)
        torchvision.utils.save_image(depth_est, os.path.join(depth_path, 'color_{0:05d}.png'.format(idx)))

        #################################
        # Construct the path for the ground truth depth image
        # ground_truth_depth_path = os.path.join(depth_GT[0], os.path.basename(model_path), name, 'ours_' + GT_iters[0], 'depth', '{0:05d}.png'.format(idx))
        rendered_depthmap = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
    
        # normalized_gt_depth = get_depth_img(ground_truth_depth_path)
        normalized__predict_depth = get_depth_img(rendered_depthmap)

        # depth_diff1 = torch.abs(normalized_gt_depth - normalized__predict_depth)
        # depth_diff2 = torch.abs(normalized_gt_depth - (1-depth))

        # print(torch.mean(depth_diff1) ,  torch.mean(depth_diff2))

        # depth_diff = depth_diff2

        # depth_diff_normalized = (depth_diff - depth_diff.min()) / (depth_diff.max() - depth_diff.min())
        # depth_diff_normalized = depth_diff_normalized[:, 0:1, :, :].squeeze(1)

        # print(depth_diff_normalized.shape)
        # print(depth.shape)
        # Save depth difference map
        # torchvision.utils.save_image(depth_diff_normalized, os.path.join(depth_path, 'depth_diff_bw_{0:05d}.png'.format(idx)))
        # depth_diff_normalized_est = depth_diff_normalized.squeeze().cpu().numpy()
        # depth_diff_normalized_est = visualize_cmap(depth_diff_normalized_est, np.ones_like(depth_diff_normalized_est), cm.get_cmap('turbo'), curve_fn=depth_diff_curve).copy()
        #print(depth_diff_normalized_est.shape)
        # depth_diff_normalized_est = torch.as_tensor(depth_diff_normalized_est).permute(2, 0, 1)
        # torchvision.utils.save_image(depth_diff_normalized_est, os.path.join(depth_path, 'depth_diff_{0:05d}.png'.format(idx)))
        ######################


    # for idx, view in enumerate(pbar):
    #     render_pkg = render(view, gaussians, pipeline, background) #render(view, gaussians, pipeline, background)["render"]
    #     rendering = render_pkg["render"]
    #     gt = view.original_image[0:3, :, :]
    #     depth = (render_pkg['depth'] - render_pkg['depth'].min()) / (render_pkg['depth'].max() - render_pkg['depth'].min()) + 1 * (1 - render_pkg["alpha"])
    #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    #     torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    #     torchvision.utils.save_image(1 - depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
    #     torchvision.utils.save_image(render_pkg["alpha"], os.path.join(depth_path, 'alpha_{0:05d}'.format(idx) + ".png"))

    #     depth_est = depth.squeeze().cpu().numpy()
    #     depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
    #     depth_est = torch.as_tensor(depth_est).permute(2,0,1)
    #     torchvision.utils.save_image(depth_est, os.path.join(depth_path, 'color_{0:05d}'.format(idx) + ".png"))

    #     # depth_map = vis_depth(render_pkg['depth'].squeeze().cpu().numpy())
    #     # np.save(os.path.join(render_path, view.image_name + '_depth.npy'), render_pkg['depth'][0].detach().cpu().numpy())
    #     # cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)

    #     if (True): #Set to false if you don't want to generate the heatmap
    #         ground_truth_depthmap = os.path.join(depth_GT[0], os.path.basename(model_path), name, 'ours_'+GT_iters[0], 'depth', '{0:05d}'.format(idx) + ".png")
    #         rendered_depthmap = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
    #         # print("ground_truth_depthmap: ", ground_truth_depthmap)
    #         # print("rendered_depthmap: ", rendered_depthmap)
    #         ground_truth = cv2.imread(ground_truth_depthmap, cv2.IMREAD_UNCHANGED)
    #         rendered_map = cv2.imread(rendered_depthmap, cv2.IMREAD_UNCHANGED)
    #         assert ground_truth.shape == rendered_map.shape, "Depth maps must be of the same shape."
    #         abs_diff = np.abs(ground_truth - rendered_map)
    #         # Create the heatmap
    #         plt.figure(figsize=(10, 8))
    #         heatmap = plt.imshow(abs_diff, cmap='hot', interpolation='nearest')
    #         plt.colorbar(heatmap, label='Absolute Difference')
    #         plt.savefig(os.path.join(depth_path, 'depth_diff_heatmap_{0:05d}'.format(idx) + ".png"), bbox_inches='tight', dpi=300)

    testing_time = time.strftime("%H:%M:%S", time.gmtime(pbar.format_dict['elapsed']))
    #testing_time = pbar.format_dict['elapsed']
    #print(pbar)
    try:
    # Try to open the file for reading
        with open('full_logs.json', "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize existing_data as an empty dictionary
        existing_data = {}
        
    #print("MOODel path",model_path)
    
    # Update testing time for the model path
    model_size = os.path.getsize(model_path+'/point_cloud/iteration_'+str(iteration)+'/point_cloud.ply') / (1024 ** 2)
    if model_path in existing_data:
        existing_data[model_path]["testing_time"] = testing_time
        existing_data[model_path]["model_size"] = model_size
    else:
        existing_data[model_path] = {"testing_time": testing_time}
        existing_data[model_path] = {"model_size": model_size}
    # Write the updated data back to the file
    with open('full_logs.json', "w") as file:
        json.dump(existing_data, file, indent=4)

############################

def render_set_old(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    pbar = tqdm(views, desc="Rendering progress")
    for idx, view in enumerate(pbar):
        rendering = render(view, gaussians, pipeline, background)["render"]
       # print("Looking like a wow == ",rendering)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    #print("Looking like a wow == ",pbar.format_dict['elapsed'])
    testing_time = time.strftime("%H:%M:%S", time.gmtime(pbar.format_dict['elapsed']))
    #testing_time = pbar.format_dict['elapsed']
    #print(pbar)
    try:
    # Try to open the file for reading
        with open('full_logs.json', "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize existing_data as an empty dictionary
        existing_data = {}
        
    #print("MOODel path",model_path)
    
    # Update testing time for the model path
    model_size = os.path.getsize(model_path+'/point_cloud/iteration_'+str(iteration)+'/point_cloud.ply') / (1024 ** 2)
    if model_path in existing_data:
        existing_data[model_path]["testing_time"] = testing_time
        existing_data[model_path]["model_size"] = model_size
    else:
        existing_data[model_path] = {"testing_time": testing_time}
        existing_data[model_path] = {"model_size": model_size}
    # Write the updated data back to the file
    with open('full_logs.json', "w") as file:
        json.dump(existing_data, file, indent=4)


def render_video(source_path, model_path, iteration, views, gaussians, pipeline, background, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = copy.deepcopy(views[0])

    if source_path.find('llff') != -1:
        # render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
        render_poses = generate_spiral_path(views, np.load(source_path + '/poses_bounds.npy'))
    elif source_path.find('360') != -1:
        render_poses = generate_ellipse_path(views)
    render_poses = generate_ellipse_path(views)
    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)
    # final_video = cv2.VideoWriter(os.path.join('/ssd1/zehao/gs_release/video/', str(iteration), model_path.split('/')[-1] + '.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.)
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()

def render_sets(dataset : ModelParams, iterations : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, depth_GT, GT_iters):
    for iteration in iterations:
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree)

            # Saving the scales 
            # print(dataset.model_path)
            # scene_name = os.path.basename(dataset.model_path)
            # scaling_copy = gaussians._scaling.detach().clone()
            # file_path = os.path.join('/home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/scales', f"scaling_copy_{scene_name}.pt")
            # torch.save(scaling_copy, file_path)

            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            if args.video:
                render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTestCameras(),
                            gaussians, pipeline, background, args.fps)
                
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, depth_GT, GT_iters)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, depth_GT, GT_iters)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", nargs="+", type=int, default=-1)
    #parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument('--depth_GT', required=False, nargs="+", type=str, default=['/home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs'])
    parser.add_argument('--GT_iters', required=False, nargs="+", type=str, default=['30000'])
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    #args.iteration = 15000
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.depth_GT, args.GT_iters)