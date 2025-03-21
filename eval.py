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
)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
)
from flow3d.params import GaussianParams, MotionBases
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server

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
    num_bg: int = 10_000
    num_motion_bases: int = 11
    num_epochs: int = 500
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 100
    save_videos_every: int = 100
    ignore_cam_mask: int = 0

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

    val_img_datases = [train[3] for train in train_list]

    # Initialize model
    initialize_and_checkpoint_model(
        cfgs[0],
        train_datasets,
        device,
        ckpt_path,
        vis=cfgs[0].vis_debug,
        port=cfgs[0].port,
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
        )
        for i, (view, val_img_dataset) in enumerate(zip(train_video_views, val_img_datases))
    ]

    guru.info(f"Starting training from {trainer.global_step=}")

    validator.save_train_videos(cfgs[0].num_epochs)
    for ind, validator in enumerate(validators):
      val_logs = validator.validate()
      metrics_str = "\n".join([f"{key}: {value}" for key, value in val_logs.items()])
      with open(f"{cfg.work_dir}/validation_metrics_cam{ind}.txt", "a") as f:  
          f.write(f"Epoch {cfgs[0].num_epochs}\n")
          f.write(metrics_str + "\n\n")

def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_datasets: list[BaseDataset],
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
    debug=False
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return

    Ks_fuse = []
    w2cs_fuse = []

    fg_params, motion_bases, bg_params, tracks_3d = init_model_from_unified_tracks(
        train_datasets,
        cfg.num_fg,
        cfg.num_bg,
        cfg.num_motion_bases,
        vis=vis,
        port=port,
    )
    for train_dataset in train_datasets:
        Ks = train_dataset.get_Ks().to(device)
        w2cs = train_dataset.get_w2cs().to(device)
        Ks_fuse.append(Ks)
        w2cs_fuse.append(w2cs)

    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs, num_iters=1)

    Ks_fuse = torch.cat(Ks_fuse, dim=0)  # Flatten [N, Ks] to [N * Ks]
    w2cs_fuse = torch.cat(w2cs_fuse, dim=0)  # Flatten w2cs similarly
    if vis and cfg.port is not None:
        server = get_server(port=cfg.port)
        vis_init_params(server, fg_params, motion_bases)

    model = SceneModel(Ks, w2cs, fg_params, motion_bases, bg_params)

    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)

def init_model_from_unified_tracks(
    train_datasets,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int,
    vis: bool = False,
    port: int | None = None,
):
    # Prepare lists to collect data from each dataset
    tracks_3d_list = []
    visibles_list = []
    invisibles_list = []
    confidences_list = []
    colors_list = []
    feats_list = []

    # Loop over the datasets and collect data
    for train_dataset in train_datasets:
        tracks_3d, visibles, invisibles, confidences, colors, feats = train_dataset.get_tracks_3d(num_fg)
        tracks_3d_list.append(tracks_3d)
        visibles_list.append(visibles)
        invisibles_list.append(invisibles)
        confidences_list.append(confidences)
        colors_list.append(colors)
        feats_list.append(feats)

    # Concatenate the data
    combined_tracks_3d = torch.cat(tracks_3d_list, dim=0)
    combined_visibles = torch.cat(visibles_list, dim=0)
    combined_invisibles = torch.cat(invisibles_list, dim=0)
    combined_confidences = torch.cat(confidences_list, dim=0)
    combined_colors = torch.cat(colors_list, dim=0)
    combined_feats = torch.cat(feats_list, dim=0)

    combined_data = (
        combined_tracks_3d,
        combined_visibles,
        combined_invisibles,
        combined_confidences,
        combined_colors,
        combined_feats,
    )

    tracks_3d = TrackObservations(*combined_data)

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )

    motion_bases = motion_bases.to(device)
    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points_list = []
        bg_normals_list = []
        bg_colors_list = []
        bg_feats_list = []

        for train_dataset in train_datasets:
            bg_points, bg_normals, bg_colors, bg_feats = train_dataset.get_bkgd_points(num_bg)
            bg_points_list.append(bg_points)
            bg_normals_list.append(bg_normals)
            bg_colors_list.append(bg_colors)
            bg_feats_list.append(bg_feats)

        combined_bg_points = torch.cat(bg_points_list, dim=0)
        combined_bg_normals = torch.cat(bg_normals_list, dim=0)
        combined_bg_colors = torch.cat(bg_colors_list, dim=0)
        combined_bg_feats = torch.cat(bg_feats_list, dim=0)

        combined_bg_data = (
            combined_bg_points,
            combined_bg_normals,
            combined_bg_colors,
            combined_bg_feats,
        )

        bg_points = StaticObservations(*combined_bg_data)
        assert bg_points.check_sizes()
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, bg_params, tracks_3d

if __name__ == "__main__":
    import wandb

    import argparse
    work_dir = './results_dance/output_noC_noO_rd_dancing_w_track_generaltest_/'
    wandb.init(name=work_dir.split('/')[-1])
    parser = argparse.ArgumentParser(description="Wandb Training Script")
    parser.add_argument(
        "--train_indices", type=int, nargs='+', default=[0, 1, 2, 3], help="List of training indices"
    )
    parser.add_argument(
        "--work_dir_name", type=str, default="complete"
    )

    args, remaining_args = parser.parse_known_args()

    train_indices = args.train_indices
    def find_missing_number(nums):
        full_set = {0, 1, 2, 3}
        missing_number = full_set - set(nums)
        return missing_number.pop() if missing_number else None

    if len(train_indices) != 4:
      work_dir += f'roll_out_cam_{find_missing_number(train_indices)}'
    
    print(work_dir)
    import tyro
    configs = [
        TrainConfig(
            work_dir=work_dir,
            data=CustomDataConfig(
                seq_name=f"undist_cam0{i+1}",
                root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
                video_name='_dance'
            ),
            # Pass the unknown arguments to tyro.cli
            lr=tyro.cli(SceneLRConfig, args=remaining_args),
            loss=tyro.cli(LossesConfig, args=remaining_args),
            optim=tyro.cli(OptimizerConfig, args=remaining_args),
            train_indices=train_indices
        )
        for i in range(4)
    ]
    main(configs)