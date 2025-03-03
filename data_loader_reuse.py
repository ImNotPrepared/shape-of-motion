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

def depth_vis(depth_np, colored_png_path, d_min=0.1, d_max=10.0):
  import cv2
  # d_min, d_max = depth_np.min(), depth_np.max()
  if d_max == d_min:
      depth_scaled = np.zeros_like(depth_np, dtype=np.uint8)
  else:
      depth_scaled = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)

  depth_map_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
  cv2.imwrite(colored_png_path, depth_map_colored)

  depth_map_rgb = cv2.cvtColor(depth_map_colored, cv2.COLOR_BGR2RGB)

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
            batch_size=cfg.batch_size,
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
        'moge': 
    }
    depth_type = 'moge'
    for batches in zip(*train_loaders):
        gt_depth = batches['depth'] 
        masks = batches['mask']
        mono_depth = os.path.join(depth_configs[depth_type], )
        depth_vis(depth_np, colored_png_path, d_min=depth_np.min(), d_max=depth_np.max())

        
         # [to_device(batch, device) for batch in batches]






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
        '1':3,
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
          for i in range(4)
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