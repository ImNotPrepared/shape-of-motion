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
    SynchornizedDataset,
    EgoDataset,
    StatDataset
)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_stat_bg, 
    init_bg_from_depth,
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
    num_epochs: int = 5000
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 4
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
    num_epochs: int = 490
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
    train_stat_list = []
    seq = 'cmu_bike'
    import json

    t, md = 0, json.load(open(f"/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{seq}/train_meta_f670.json", 'r'))
    for cfg in cfgs[:1]:
        train_dataset = EgoDataset(t=t, md=md, seq=seq)
        train_stat_dataset = StatDataset(t=t, md=md, seq=seq)
        # train_dataset = StatDataset(t=t, md=md, seq=seq)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(train_dataset)
        # save config
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=False)
        ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_dl_workers,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn,
        )

        train_stat_loader = DataLoader(
            train_stat_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_dl_workers,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn,
        )

        train_list.append((train_loader, train_loader, train_dataset, ))
        train_stat_list.append((train_stat_loader, train_stat_loader, train_stat_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders = [train[1] for train in train_list]
    train_datasets = [train[2] for train in train_list]

    train_stat_loaders = [train[1] for train in train_stat_list]
    train_stat_datasets = [train[2] for train in train_stat_list]


    initialize_and_checkpoint_model(
        cfgs[0],
        train_stat_datasets,
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

    guru.info(f"Starting training from {trainer.global_step=}")
    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfgs[0].num_epochs),
            initial=start_epoch,
            total=cfgs[0].num_epochs,
        )
    ):
        loss = 0
        trainer.set_epoch(epoch)
        print('?????? training')
        for batches, stat_batches in zip(zip(*train_loaders), zip(*train_stat_loaders)):
            print('Indeed training')
            batches = [to_device(batch, device) for batch in batches]
            stat_batches = [to_device(batch, device) for batch in stat_batches]
            loss = trainer.train_stat_step(stat_batches, None)# batches, 
            loss.backward()
            trainer.op_af_bk()
            pbar.set_description(f"Loss: {loss:.6f}")

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

    Ks_fuse = []
    w2cs_fuse = []

    bg_params = init_stat_model(
    train_datasets,
    num_fg=0,
    num_bg=cfg.num_bg
    )

    train_dataset = train_datasets[0]
    bg_points = StaticObservations(*train_dataset.get_bkgd_points(1))
    assert bg_points.check_sizes()

    bg_params = init_bg_from_depth(bg_points)
    

  
    bg_params = bg_params.to(device)


    for train_dataset in train_datasets:
        Ks = train_dataset.get_Ks().to(device)
        w2cs = train_dataset.get_w2cs().to(device)
        Ks_fuse.append(Ks)
        w2cs_fuse.append(w2cs)

    Ks_fuse = torch.cat(Ks_fuse, dim=0)  # Flatten [N, Ks] to [N * Ks]
    w2cs_fuse = torch.cat(w2cs_fuse, dim=0)  # Flatten w2cs similarly

    model = SceneModel(Ks, w2cs, fg_params=None, motion_bases=None, bg_params=bg_params)
    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)

def init_stat_model(
    train_datasets,
    num_fg: int,
    num_bg: int,
    vis: bool = False,
    port: int | None = None,
    seq_name: str = ''
):
    bg_params = None
    bg_params = init_stat_bg(seq_name='bike')
    bg_params = bg_params.cuda()
    return bg_params

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
    else:
      configs = [
          TrainBikeConfig(
              work_dir=work_dir,
              data=CustomDataConfig(
                  seq_name=f"{category}_undist_cam0{i+1}",
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
      # /data3/zihanwa3/Capstone-DSR/raw_data/unc_basketball_03-16-23_01_18/trajectory/Dy_train_meta.json
    main(configs)