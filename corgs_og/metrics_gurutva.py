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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser


def mae2(predicted):
    """Calculate Mean Absolute Error (MAE)"""
    return torch.mean(predicted)

def mae(predicted, target):
    """Calculate Mean Absolute Error (MAE)"""
    mae = torch.mean(torch.abs(predicted - target))
    print(mae)
    return mae

def readImages(renders_dir, gt_dir, depths_diff_dir , gt_depths_diff_dir):
    renders = []
    gts = []
    depths_diff = []
    gt_depths_diff = []

    image_names = []
    #print(type(depths_diff_dir))
    # /media/user/New Volume/Gurutva/Nerf_exp/gaussian_splat/original/3dgs_2d/output/down4_3views_7000_sparsept_llff
    # /media/user/New Volume/Gurutva/Nerf_exp/gaussian_splat/original/3dgs_test/output/down4_200views_30000_sparsept_llff
    
    # gt_depths_diff_dir = Path(str(depths_diff_dir).replace("7000", "30000"))
    # gt_depths_diff_dir = Path(str(gt_depths_diff_dir).replace("3views", "200views"))
    # gt_depths_diff_dir = Path(str(gt_depths_diff_dir).replace("dense", "sparse"))
    
    for fname in os.listdir(renders_dir):
        if 'IMG' in fname or 'DJI' in fname or 'DS' in fname:
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        depth_diff = Image.open(depths_diff_dir / fname)

        # depth_diff_path = depths_diff_dir / f"depth_diff_heatmap_{fname}"
        # depth_diff = Image.open(depth_diff_path)

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        depths_diff.append(tf.to_tensor(depth_diff).unsqueeze(0)[:, :3, :, :].cuda())

        image_names.append(fname)

        gt_depth_diff = Image.open(gt_depths_diff_dir / fname)
        gt_depths_diff.append(tf.to_tensor(gt_depth_diff).unsqueeze(0)[:, :3, :, :].cuda())

        # print("predicted depth =>",depths_diff_dir / fname)
        # print("gt depth =>",gt_depths_diff_dir / fname)

    return renders, gts, image_names , depths_diff ,gt_depths_diff

def evaluate(model_paths,depth_GT,GT_iters):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            scene_name = scene_dir.split('/')[-1]
            #print(name)
            ground_truth_depth_path = Path(os.path.join(depth_GT[0], scene_name,'test', 'ours_' + GT_iters[0]))


            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                depths_diff_dir = method_dir / "depth"
                ground_truth_depth_dir = ground_truth_depth_path / "depth"
                renders, gts, image_names , depths_diff , gt_depths_diff= readImages(renders_dir, gt_dir, depths_diff_dir , ground_truth_depth_dir)

                ssims = []
                psnrs = []
                lpipss = []
                maes = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    maes.append(mae(depths_diff[idx],gt_depths_diff[idx]).item())  # Calculate MAE

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("   MAE : {:>12.7f}".format(torch.tensor(maes).mean(), ".5"))  # Print MAE
                print("")
                
                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "MAE": torch.tensor(maes).mean().item()  # Store MAE
                })
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "MAE": {name: mae for mae, name in zip(maes, image_names)}  # Store per-view MAE
                })
            try:
            # Try to open the file for reading
                with open('full_logs.json', "r") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                # If the file doesn't exist, initialize existing_data as an empty dictionary
                existing_data = {}

            # Update testing time for the model path
            model_path=scene_dir
            if model_path in existing_data:
                existing_data[model_path]["metric"] = full_dict[scene_dir]
            else:
                existing_data[model_path] = {"metric": full_dict[scene_dir]}

            # Write the updated data back to the file
            with open('full_logs.json', "w") as file:
                json.dump(existing_data, file, indent=4)
                    
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print(e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--depth_GT', required=False, nargs="+", type=str, default=['/home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/llff_3dgs'])
    parser.add_argument('--GT_iters', required=False, nargs="+", type=str, default=['30000'])
    args = parser.parse_args()
    evaluate(args.model_paths,args.depth_GT, args.GT_iters)
