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
    num_fg: int = 70_000
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
    num_fg: int = 70_000
    num_bg: int = 70_000
    num_motion_bases: int = 14
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


def interpolate_cameras(c2w1, c2w2, alpha):
    """
    Interpolates between two camera extrinsics using slerp and lerp.

    Args:
        c2w1 (np.ndarray): First camera-to-world matrix (4x4).
        c2w2 (np.ndarray): Second camera-to-world matrix (4x4).
        alphas (list or np.ndarray): List of interpolation factors between 0 and 1.

    Returns:
        list: List of interpolated camera-to-world matrices.
    """
    # Extract rotations and translations
    R1 = c2w1[:3, :3]
    t1 = c2w1[:3, 3]
    R2 = c2w2[:3, :3]
    t2 = c2w2[:3, 3]
    
    # Create Rotation objects
    key_rots = R.from_matrix([R1, R2])
    key_times = [0, 1]
    
    # Create the slerp object
    slerp = Slerp(key_times, key_rots)
    
    interpolated_c2ws = []

    # Interpolate rotation at time alpha
    r_interp = slerp([alpha])[0]
    
    # Perform lerp on translations
    t_interp = (1 - alpha) * t1 + alpha * t2
    
    # Reconstruct the c2w matrix
    c2w_interp = np.eye(4)
    c2w_interp[:3, :3] = r_interp.as_matrix()
    c2w_interp[:3, 3] = t_interp
    
    interpolated_c2ws.append(c2w_interp)
    
    return interpolated_c2ws

def main(cfgs: List[TrainConfig]):
    train_list = []
    for cfg in cfgs:
        train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
            get_train_val_datasets(cfg.data, load_val=True)
        )
        guru.info(f"Training dataset has {train_dataset.num_frames} frames")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # save config
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=False)

        # if checkpoint exists
        ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_dl_workers,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn,
        )
        
        train_indices = cfg.train_indices

        # val_img_datases

        train_list.append((train_video_view, train_loader, train_dataset, train_video_view))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = [train[1] for train in train_list]
    train_loaders = [train_loader[i] for i in train_indices]


    train_video_views = [train[0] for train in train_list]


    train_dataset = [train[2] for train in train_list]
    train_datasets = [train_dataset[i] for i in train_indices]
    print(len(train_dataset), len(train_datasets))

    syn_dataset = SynchornizedDataset(train_datasets)
    syn_dataloader = DataLoader(
            syn_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_dl_workers,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn_sync,
        )
    
    val_img_datases = [train[3] for train in train_list]

    # Initialize model
    initialize_and_checkpoint_model(
        cfgs[0],
        train_datasets,
        device,
        ckpt_path,
        vis=cfgs[0].vis_debug,
        port=cfgs[0].port,
        seq_name=cfg.seq_name
    )

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path,
        device,
        cfgs[0].lr,
        cfgs[0].loss,
        cfgs[0].optim,
        work_dir=cfgs[0].work_dir,
        port=cfgs[0].port,
    )

    validators = [
        Validator(
            model=trainer.model,
            device=device,
            train_loader=DataLoader(view, batch_size=1),
            val_img_loader=(
                DataLoader(val_img_dataset, batch_size=110) 
            ),
            save_dir=os.path.join(cfgs[0].work_dir, f'cam_{i+1}'),
            do_the_trick=0.97
        )
        for i, (view, val_img_dataset) in enumerate(zip(train_video_views, val_img_datases))
    ]
    import json
    def noisy_camera(w2c):
      """
      Interpolates between two camera extrinsics using slerp and lerp.

      Args:
          c2w1 (np.ndarray): First camera-to-world matrix (4x4).
          c2w2 (np.ndarray): Second camera-to-world matrix (4x4).
          alphas (list or np.ndarray): List of interpolation factors between 0 and 1.

      Returns:
          list: List of interpolated camera-to-world matrices.
      """
      # Extract rotations and translations
      R = w2c[:3, :3]
      t = w2c[:3, 3]

      fixed_angle_rad = np.deg2rad(5)  # Convert 5 degrees to radians
      noise_angles = np.array([fixed_angle_rad] * 3)

      # Create rotation matrices around x, y, and z axes
      Rx = np.array([
          [1, 0, 0],
          [0, np.cos(noise_angles[0]), -np.sin(noise_angles[0])],
          [0, np.sin(noise_angles[0]),  np.cos(noise_angles[0])]
      ])

      Ry = np.array([
          [ np.cos(noise_angles[1]), 0, np.sin(noise_angles[1])],
          [0, 1, 0],
          [-np.sin(noise_angles[1]), 0, np.cos(noise_angles[1])]
      ])

      Rz = np.array([
          [np.cos(noise_angles[2]), -np.sin(noise_angles[2]), 0],
          [np.sin(noise_angles[2]),  np.cos(noise_angles[2]), 0],
          [0, 0, 1]
      ])

      # Combine the rotation matrices
      R_noise = Rz @ Ry @ Rx

      # Apply the rotation noise to the original rotation
      R_new = R_noise @ R

      # Construct the new w2c matrix with the noisy rotation and original translation
      w2c_new = np.eye(4)
      w2c_new[:3, :3] = R_new
      w2c_new[:3, 3] = t

      # Update w2c with the new matrix
      w2c = w2c_new
      interpolated_c2ws = []
      interpolated_c2ws.append(w2c)
      
      return interpolated_c2ws

    import json
    try: 
        md = json.load(open(cfg.test_w2cs, 'r'))
        c2ws = []
        for c in range(1, 6):
          if c==5:
              c=1
          k, w2c =  md['k'][0][c], np.linalg.inv(md['w2c'][0][c])
          c2ws.append(w2c)


        all_interpolated_c2ws = []
        all_interpolated_c2ws_ = []
        for i in range(len(c2ws) - 1):
            c2w1 = c2ws[i]
            c2w2 = c2ws[i + 1]
            interpolated = interpolate_cameras(c2w1, c2w2, alpha=0.5)
            all_interpolated_c2ws.extend(interpolated)
            all_interpolated_c2ws_.extend(noisy_camera(c2w1))

    except:
        md = json.load(open(cfg.test_w2cs.replace('raw_data/', 'Processing_'), 'r'))
        c2ws = []
        cam_ids = [3, 21, 23, 25]
        for i, kv in enumerate(cam_ids):
            if kv > 30:
                k = 4
            elif kv > 15:
                k = 3
            elif kv > 10:
                k = 2
            else:
                k = 1

            cam_ids[i] = kv - k 

        intrinsics = [np.array(md['k'][0][i]) for i in cam_ids]
        w2cs = [np.array(md['w2c'][0][i]) for i in cam_ids]
        
        # k, w2c =  md['k'][0][c], np.linalg.inv(md['w2c'][0][c])
        c2ws = w2cs#.tolist()]
        c2ws.append(c2ws[0])
        
        all_interpolated_c2ws = []
        all_interpolated_c2ws_ = []
        for i in range(len(c2ws) - 1):
            c2w1 = c2ws[i]
            c2w2 = c2ws[i + 1]
            interpolated = interpolate_cameras(c2w1, c2w2, alpha=0.5)
            all_interpolated_c2ws.extend(interpolated)
            all_interpolated_c2ws_.extend(noisy_camera(c2w1))


    guru.info(f"Starting training from {trainer.global_step=}")
    
    epoch = trainer.global_step
    for iiidx, validator in enumerate(validators):
        validator.save_train_videos_images(epoch)
        validator.save_int_videos(epoch, all_interpolated_c2ws[iiidx])

    '''import collections

    all_val_logs = collections.defaultdict(list)

    # Iterate over validators and collect metrics
    for ind, validator in enumerate(validators):
        val_logs = validator.validate()
        metrics_str = "\n".join([f"{key}: {value}" for key, value in val_logs.items()])

        # Append metrics to the list for calculating means
        for key, value in val_logs.items():
            all_val_logs[key].append(value)

        # Write individual metrics to file
        with open(f"{cfg.work_dir}/NEW_validation_metrics_cam{ind}.txt", "a") as f:
            f.write(f"Epoch {cfgs[0].num_epochs}\n")
            f.write(metrics_str + "\n\n")
            
    mean_val_logs = {key: sum(values) / len(values) for key, values in all_val_logs.items()}
    mean_metrics_str = "\n".join([f"{key}: {value}" for key, value in mean_val_logs.items()])

    with open(f"{cfg.work_dir}/NEW_validation_metrics_mean.txt", "a") as f:
        f.write(f"Epoch {cfgs[0].num_epochs}\n")
        f.write(mean_metrics_str + "\n\n")
    for iiidx, validator in enumerate(validators):
        validator.save_int_videos(epoch, all_interpolated_c2ws[iiidx])
        validator.save_train_videos_images(epoch)'''



def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_datasets: list[BaseDataset],
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
    debug=False,
    seq_name: str = ''
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return



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
    elif 'panoptic' in seq_name:
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