import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Annotated

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data import (
    BaseDataset,
    CustomDataConfig,
    get_train_val_datasets,
    SynchornizedDataset
)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
    init_fg_motion_bases_from_single_t,
)
from flow3d.params import GaussianParams, MotionBases
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server
from scipy.spatial.transform import Slerp, Rotation as R
torch.set_float32_matmul_precision("high")

import numpy as np
import matplotlib


def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.999)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp), 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    return colored


def colorize_depth_affine(depth: np.ndarray, mask: np.ndarray = None, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        depth = np.where(mask, depth, np.nan)

    min_depth, max_depth = np.nanquantile(depth, 0.001), np.nanquantile(depth, 0.999)
    depth = (depth - min_depth) / (max_depth - min_depth)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](depth), 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    return colored


def colorize_disparity(disparity: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        disparity = np.where(mask, disparity, np.nan)
    
    if normalize:
        min_disp, max_disp = np.nanquantile(disparity, 0.001), np.nanquantile(disparity, 0.999)
        disparity = (disparity - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disparity), 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    return colored


def colorize_segmentation(segmentation: np.ndarray, cmap: str = 'Set1') -> np.ndarray:
    colored = matplotlib.colormaps[cmap]((segmentation % 20) / 20)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    return colored


def colorize_normal(normal: np.ndarray) -> np.ndarray:
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal


def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

@dataclass
class TrainConfig:
    work_dir: str
    train_indices: list
    data: CustomDataConfig
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 0
    num_bg: int = 14_000
    num_motion_bases: int = 21
    num_epochs: int = 500
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 100
    save_videos_every: int = 70
    ignore_cam_mask: int = 0
    test_validator_every: int = 1
    test_w2cs: str = ''
    seq_name: str = ''

@dataclass
class TrainBikeConfig:
    work_dir: str
    train_indices: list
    data: CustomDataConfig
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 21_000
    num_bg: int = 14_000
    num_motion_bases: int = 28
    num_epochs: int = 500
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 100
    save_videos_every: int = 70
    ignore_cam_mask: int = 0
    test_w2cs: str = ''
    seq_name: str = ''

def depth_vis(depth_np, colored_png_path, d_min=0.1, d_max=5.0):
  import cv2
  old = False
  if old: 
    # d_min, d_max = depth_np.min(), depth_np.max()
    if d_max == d_min:
        depth_scaled = np.zeros_like(depth_np, dtype=np.uint8)
    else:
        depth_scaled = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    depth_map_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
    cv2.imwrite(colored_png_path, depth_map_colored)

    depth_map_rgb = cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB)
  else:
    depth_map_rgb = cv2.cvtColor(colorize_depth(depth_np), cv2.COLOR_RGB2BGR)
    cv2.imwrite(colored_png_path, depth_map_rgb)

  return Image.fromarray(depth_map_rgb)

def main(cfgs: List[TrainConfig]):
    train_list = []
    for cfg in cfgs:
        train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
            get_train_val_datasets(cfg.data, load_val=True)
        )
        guru.info(f"Training dataset has {train_dataset.num_frames} frames")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            num_workers=cfg.num_dl_workers,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn,
        )
        
        train_indices = cfg.train_indices

        train_list.append((train_video_view, train_loader, train_dataset, train_video_view))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = [train[1] for train in train_list]
    train_loaders = [train_loader[i] for i in train_indices]
    train_video_views = [train[0] for train in train_list]
    train_dataset = [train[2] for train in train_list]
    train_datasets = [train_dataset[i] for i in train_indices]

    depth_configs = {
        'da2': None, 
        'moge': None, 
    }
    depth_type = 'moge'
    from moge.model import MoGeModel
    import numpy as np
    import utils3d
    device = torch.device("cuda")
    import os 

    # Load the model from huggingface hub (or load from local).
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)    
    base_path = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/moge_depth/panoptic'
    for c, batches in enumerate(zip(*train_loaders)):
        batcheses = [to_device(batch, device) for batch in batches]


        for c, batches in enumerate(batcheses):
            frames_for_gif_pure = []
            frames_for_gif = []
            
            for t in range(len(batches['imgs'])):

                #batches = batcheses[t]
                print(batches.keys())
                img = batches['imgs'][t]
                gt_depth = batches['depths'][t] 
                dyn_mask = batches['masks'][t]
                output = model.infer(img.permute(2, 0, 1))

                points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
                normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
                mask=mask & ~(utils3d.numpy.depth_edge(depth, rtol=0.03, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask))
                depth = output['depth'].detach().cpu().numpy()

                gt_depth = gt_depth.detach().cpu().numpy()
                dyn_mask = dyn_mask.detach().cpu().numpy().astype(bool)

                metric_disp_map, mono_disp_map = gt_depth, depth
                ms_colmap_disp = metric_disp_map - np.median(metric_disp_map) + 1e-8
                ms_mono_disp = mono_disp_map - np.median(mono_disp_map) + 1e-8
                scale = np.median(ms_colmap_disp / ms_mono_disp)

                depth = scale * depth

                os.makedirs(os.path.join(base_path, str(c)), exist_ok=True)
                os.makedirs(os.path.join(base_path, str(c)), exist_ok=True)

                np.save(os.path.join(base_path, str(c), f'raw_depth_{t}.npy'), depth)

                save_path = os.path.join(base_path, str(c), f'raw_depth_{t}.png')
                to_append = depth_vis(depth, save_path)

                frames_for_gif.append(to_append)
                ### dyn_mask
                #   dyn_mask
                ### dyn_mask
                filter_depth = np.where(~dyn_mask, 0, depth)
                np.save(os.path.join(base_path, str(c), f'fg_depth_{t}.npy'), filter_depth)
                save_path = os.path.join(base_path, str(c), f'fg_depth_{t}.png')
                to_append = depth_vis(filter_depth, save_path)
                frames_for_gif_pure.append(to_append)
            if len(frames_for_gif) > 0:
                gif_path = os.path.join(os.path.join(base_path, str(c)), f"_depth.gif")
                frames_for_gif[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames_for_gif[1:],
                    duration=100,  # ms per frame
                    loop=0
                )


            if len(frames_for_gif_pure) > 0:
                gif_path = os.path.join(os.path.join(base_path, str(c)), f"_depth_pure.gif")
                frames_for_gif_pure[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames_for_gif_pure[1:],
                    duration=100,  # ms per frame
                    loop=0
                )



if __name__ == "__main__":
    import wandb
    import argparse


    parser = argparse.ArgumentParser(description="Wandb Training Script")
    parser.add_argument(
        "--seq_name", type=str, default="complete"
    )
    parser.add_argument(
        "--train_indices", type=int, nargs='+', default=[0, 1, 2, 3], help="List of training indices"
    )

    parser.add_argument(
        "--exp", type=str, 
    )

    parser.add_argument(
        "--depth_type", type=str, default="monst3r+dust3r"
    )

    args, remaining_args = parser.parse_known_args()

    train_indices = args.train_indices
    seq_name = args.seq_name
    depth_type = args.depth_type

    category = seq_name.split("_")[2]

    
    work_dir = f'./results{seq_name}/{args.exp}/'
    wandb.init(name=work_dir.split('/')[-1])
    '''
    def load_depth(self, index) -> torch.Tensor:
    #  load_da2_depth load_duster_depth load_org_depth
    if self.depth_type == 'modest':
        depth = self.load_modest_depth(index)
    elif self.depth_type == 'da2':
        depth = self.load_da2_depth(index)
    elif self.depth_type == 'dust3r':
        depth = self.load_modest_depth(index)       
    elif self.depth_type == 'monst3r':
        depth = self.load_monster_depth(index)    
    elif self.depth_type == 'monst3r+dust3r':
        depth = self.load_duster_moncheck_depth(index) 
    
    '''

    def find_missing_number(nums):
        full_set = {0, 1, 2, 3}
        missing_number = full_set - set(nums)
        return missing_number.pop() if missing_number else None

    if len(train_indices) != 4:
      work_dir += f'roll_out_cam_{find_missing_number(train_indices)}'
    
    print(work_dir)
    import tyro
    if seq_name == 'dance':
      configs = [
          TrainConfig(
              work_dir=work_dir,
              data=CustomDataConfig(
                  seq_name=f"{category}_undist_cam0{i+1}",
                  root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
                  video_name=seq_name,
                  depth_type=depth_type,
              ),
              # Pass the unknown arguments to tyro.cli
              lr=tyro.cli(SceneLRConfig, args=remaining_args),
              loss=tyro.cli(LossesConfig, args=remaining_args),
              optim=tyro.cli(OptimizerConfig, args=remaining_args),
              train_indices=train_indices,
              test_w2cs=f'/data3/zihanwa3/Capstone-DSR/raw_data/{seq_name[1:]}/trajectory/Dy_train_meta.json',
              seq_name=seq_name
          )
          for i in range(4)
      ]
    elif 'panoptic' in depth_type:
      cam_dict = {
        '1': 21,
        '2':21,
        '3':23,
        '4':25,
      }
      configs = [
          TrainBikeConfig(
              work_dir=work_dir,
              data=CustomDataConfig(
                  seq_name=f"{category}_undist_cam{cam_dict[str(i+1)]:02d}",
                  root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
                  video_name=seq_name,
                  depth_type=depth_type,
                  super_fast=False
              ),
              # Pass the unknown arguments to tyro.cli
              lr=tyro.cli(SceneLRConfig, args=remaining_args),
              loss=tyro.cli(LossesConfig, args=remaining_args),
              optim=tyro.cli(OptimizerConfig, args=remaining_args),
              train_indices=train_indices,
              test_w2cs=f'/data3/zihanwa3/Capstone-DSR/raw_data/{seq_name[1:]}/trajectory/Dy_train_meta.json',
              seq_name=seq_name
          )
          for i in range(len(cam_dict))
      ]     

      ### panoptic testing data:
      # real: [1, 3, 8, 13, 19, 21] (start with 0)
      # test_real [0, 10, 15, 30]
      # amend_wrong: [0, 2, 7, 11, 16, 18]
      # amend_correct: [2, 4, 9, 11, 16, 18]
      # /data3/zihanwa3/Capstone-DSR/raw_data/unc_basketball_03-16-23_01_18/trajectory/Dy_train_meta.json
    else:
      cam_dict = {
        '1':1,
        '2':2,
        '3':3,
        '4':4,
      }
      configs = [
          TrainBikeConfig(
              work_dir=work_dir,
              data=CustomDataConfig(
                  seq_name=f"{category}_undist_cam{cam_dict[str(i+1)]:02d}",
                  root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
                  video_name=seq_name,
                  depth_type=depth_type,
                  super_fast=False
              ),
              # Pass the unknown arguments to tyro.cli
              lr=tyro.cli(SceneLRConfig, args=remaining_args),
              loss=tyro.cli(LossesConfig, args=remaining_args),
              optim=tyro.cli(OptimizerConfig, args=remaining_args),
              train_indices=train_indices,
              test_w2cs=f'/data3/zihanwa3/Capstone-DSR/raw_data/{seq_name[1:]}/trajectory/Dy_train_meta.json',
              seq_name=seq_name
          )
          for i in range(4)
      ]           
    main(configs)