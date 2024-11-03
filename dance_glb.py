import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated

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
    DavisDataConfig,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
    vis_tracks_2d_video, 
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
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
    )
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 150_000
    num_bg: int = 100_000 ### changed to 0 # 100_000
    num_motion_bases: int = 11
    num_epochs: int = 500
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 100
    save_videos_every: int = 100


def main(cfgs):
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

      train_list.append((train_video_view, train_loader, train_dataset))



    train_dataset_0 = train_list[0][-1]#.train_step(batch_0)
    train_dataset_1 = train_list[1][-1]#.train_step(batch_1)
    train_dataset_2 = train_list[2][-1]#.train_step(batch_2)
    train_dataset_3 = train_list[3][-1]#.train_step(batch_3)
    #train_dataset_4 = train_list[4][-1]#.train_step(batch_3)
    debug=False


    if debug:
      initialize_and_checkpoint_model(
          cfg,
          [train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3 ],
          device,
          ckpt_path,
          vis=cfg.vis_debug,
          port=cfg.port,
          debug=debug
      )

    else:

      initialize_and_checkpoint_model(
          cfg,
          [train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3],
          device,
          ckpt_path,
          vis=cfg.vis_debug,
          port=cfg.port,
          debug=debug
      )

    trainer, start_epoch = Trainer.init_from_checkpoint(
        ckpt_path,
        device,
        cfg.lr,
        cfg.loss,
        cfg.optim,
        work_dir=cfg.work_dir,
        port=cfg.port,
    )


    train_loader_0 = train_list[0][1]
    train_loader_1 = train_list[1][1]
    train_loader_2 = train_list[2][1]
    train_loader_3 = train_list[3][1]
    #train_loader_4 = train_list[4][1]

    train_video_view_0 = train_list[0][0]
    train_video_view_1 = train_list[1][0]
    train_video_view_2 = train_list[2][0]
    train_video_view_3 = train_list[3][0]
    #train_video_view_4 = train_list[4][0]
    
    validator_0 = Validator(
        model=trainer.model,
        device=device,
        train_loader=(
            DataLoader(train_video_view_0, batch_size=1)
        ),
        save_dir=os.path.join(cfg.work_dir, 'cam_1'),
    )

    validator_1 = Validator(
        model=trainer.model,
        device=device,
        train_loader=(
            DataLoader(train_video_view_1, batch_size=1)
        ),
        save_dir=os.path.join(cfg.work_dir, 'cam_2'),
    )

    validator_2 = Validator(
        model=trainer.model,
        device=device,
        train_loader=(
            DataLoader(train_video_view_2, batch_size=1)
        ),
        save_dir=os.path.join(cfg.work_dir, 'cam_3'),
    )
    validator_3 = Validator(
        model=trainer.model,
        device=device,
        train_loader=(
            DataLoader(train_video_view_3, batch_size=1)
        ),
        save_dir=os.path.join(cfg.work_dir, 'cam_4'),
    )


    guru.info(f"Starting training from {trainer.global_step=}")
    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfg.num_epochs),
            initial=start_epoch,
            total=cfg.num_epochs,
        )
    ):
        loss = 0


        trainer.set_epoch(epoch)

        train_loaders = [train_loader_0, train_loader_1, train_loader_2, train_loader_3]

        # Zip the loaders to load one batch from each loader at each step
        for batch_0, batch_1, batch_2, batch_3 in zip(*train_loaders):

            #### load multi-view data
            batch_0 = to_device(batch_0, device)
            batch_1 = to_device(batch_1, device)
            batch_2 = to_device(batch_2, device)
            batch_3 = to_device(batch_3, device)
            #batch_4 = to_device(batch_4, device)

            
            if debug:
              loss = trainer.train_step([batch_0, batch_0, batch_0, batch_0])

            else:
              loss = trainer.train_step([batch_0, batch_1, batch_2, batch_3])#(loss_0 + loss_1 + loss_2 + loss_3) / 4
            loss.backward()
            trainer.op_af_bk()

            pbar.set_description(f"Loss: {loss:.6f}")


        if (epoch > 0 and epoch % cfg.save_videos_every == 0) or (
            epoch == cfg.num_epochs - 1
        ):
          validator_0.save_train_videos(epoch)
          validator_1.save_train_videos(epoch)
          validator_2.save_train_videos(epoch)
          validator_3.save_train_videos(epoch)
          #validator_4.save_train_videos(epoch)



def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_datasets: list[BaseDataset], #train_dataset: BaseDataset, 
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
    fg_params_fuse = []
    motion_bases_fuse = []
    bg_params_fuse = []

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

    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs, num_iters=1122)

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
    
    for idx, train_dataset in enumerate(train_datasets[:1]):
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



def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["flow3d", "scripts"]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))


if __name__ == "__main__":
    import wandb 
    import argparse
    import tyro

    wandb.init()  

    work_dir = './results_dance/output_noC_largeD_nodisp_rd_dancing_w_track'
    config_1 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="undist_cam01",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
            video_name='_dance'
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_2 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="undist_cam02",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
            video_name='_dance'
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_3 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="undist_cam03",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
            video_name='_dance'
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_4 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="undist_cam04",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
            video_name='_dance'
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_5 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="undist_cam05",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
            video_name='_dance'
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    main([config_1, config_2, config_3, config_4])