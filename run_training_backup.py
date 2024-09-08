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
)
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
    num_fg: int = 40_000
    num_bg: int = 100_000
    num_motion_bases: int = 10
    num_epochs: int = 500
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50


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
      initialize_and_checkpoint_model(
          cfg,
          train_dataset,
          device,
          ckpt_path,
          vis=cfg.vis_debug,
          port=cfg.port,
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

      train_loader = DataLoader(
          train_dataset,
          batch_size=cfg.batch_size,
          num_workers=cfg.num_dl_workers,
          persistent_workers=True,
          collate_fn=BaseDataset.train_collate_fn,
      )


      validator = None
      if (
          train_video_view is not None
          or val_img_dataset is not None
          or val_kpt_dataset is not None
      ):
          validator = Validator(
              model=trainer.model,
              device=device,
              train_loader=(
                  DataLoader(train_video_view, batch_size=1) if train_video_view else None
              ),
              val_img_loader=(
                  DataLoader(val_img_dataset, batch_size=1) if val_img_dataset else None
              ),
              val_kpt_loader=(
                  DataLoader(val_kpt_dataset, batch_size=1) if val_kpt_dataset else None
              ),
              save_dir=cfg.work_dir,
          )
      train_list.append((trainer , train_loader, validator))

    guru.info(f"Starting training from {trainer.global_step=}")

    trainer_0 = train_list[0][0]#.train_step(batch_0)
    trainer_1 = train_list[1][0]#.train_step(batch_1)
    trainer_2 = train_list[2][0]#.train_step(batch_2)
    trainer_3 = train_list[3][0]#.train_step(batch_3)

    train_loader_0 = train_list[0][1]
    train_loader_1 = train_list[1][1]
    train_loader_2 = train_list[2][1]
    train_loader_3 = train_list[3][1]


    for epoch in (
        pbar := tqdm(
            range(start_epoch, cfg.num_epochs),
            initial=start_epoch,
            total=cfg.num_epochs,
        )
    ):
        loss = 0


        trainer_0.set_epoch(epoch)
        trainer_1.set_epoch(epoch)
        trainer_2.set_epoch(epoch)
        trainer_3.set_epoch(epoch)


        train_loaders = [train_loader_0, train_loader_1, train_loader_2, train_loader_3]

        # Zip the loaders to load one batch from each loader at each step
        for batch_0, batch_1, batch_2, batch_3 in zip(*train_loaders):

            #### load multi-view data
            batch_0 = to_device(batch_0, device)
            batch_1 = to_device(batch_1, device)
            batch_2 = to_device(batch_2, device)
            batch_3 = to_device(batch_3, device)

            loss_0 = trainer_0.train_step(batch_0)
            loss_1 = trainer_1.train_step(batch_1)
            loss_2 = trainer_2.train_step(batch_2)
            loss_3 = trainer_3.train_step(batch_3)


            #batch = to_device(batch, device)
            #loss = trainer.train_step(batch)

            loss = (loss_0 + loss_1 + loss_2 + loss_3) / 4
            loss.backward()

            trainer_0.op_af_bk()
            trainer_1.op_af_bk()
            trainer_2.op_af_bk()
            trainer_3.op_af_bk()

            pbar.set_description(f"Loss: {loss:.6f}")

        for _, _, validator in train_list:
          if validator is not None:
              if (epoch > 0 and epoch % cfg.validate_every == 0) or (
                  epoch == cfg.num_epochs - 1
              ):
                  val_logs = validator.validate()
                  trainer.log_dict(val_logs)
              if (epoch > 0 and epoch % cfg.save_videos_every == 0) or (
                  epoch == cfg.num_epochs - 1
              ):
                  validator.save_train_videos(epoch)


def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_datasets: list[BaseDataset], #train_dataset: BaseDataset, 
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return
    

    Ks_fuse = []
    w2cs_fuse = []
    fg_params_fuse = []
    motion_bases_fuse = []
    bg_params_fuse = []

    for train_dataset in train_datasets:
        # Initialize model from tracks
        fg_params, motion_bases, bg_params, tracks_3d = init_model_from_tracks(
            train_dataset,
            cfg.num_fg,
            cfg.num_bg,
            cfg.num_motion_bases,
            vis=vis,
            port=port,
        )
        # Get camera intrinsic matrices and world-to-camera transformations
        Ks = train_dataset.get_Ks().to(device)
        w2cs = train_dataset.get_w2cs().to(device)


        run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs)
        Ks_fuse.append(Ks)
        w2cs_fuse.append(w2cs)
        fg_params_fuse.append(fg_params)
        motion_bases_fuse.append(motion_bases)
        bg_params_fuse.append(bg_params)
        

    Ks_fuse = torch.cat(Ks_fuse, dim=0)  # Flatten [N, Ks] to [N * Ks]
    w2cs_fuse = torch.cat(w2cs_fuse, dim=0)  # Flatten w2cs similarly
    fg_params_fuse = torch.cat(fg_params_fuse, dim=0)  # Flatten fg_params
    motion_bases_fuse = torch.cat(motion_bases_fuse, dim=0)  # Flatten motion_bases
    bg_params_fuse = torch.cat(bg_params_fuse, dim=0)  # Flatten bg_params

    model = SceneModel(Ks_fuse, w2cs_fuse, fg_params_fuse, motion_bases_fuse, bg_params_fuse)


    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)


def init_model_from_tracks(
    train_dataset,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int,
    vis: bool = False,
    port: int | None = None,
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    print(
        f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
        f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape} "
        f"{tracks_3d.colors.shape}"
    )
    if not tracks_3d.check_sizes():
        import ipdb

        ipdb.set_trace()

    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())
    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
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
    config_1 = TrainConfig(
        work_dir="./outdir",
        data=CustomDataConfig(
            seq_name="toy_512_1",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_2 = TrainConfig(
        work_dir="./outdir",
        data=CustomDataConfig(
            seq_name="toy_512_2",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_3 = TrainConfig(
        work_dir="./outdir",
        data=CustomDataConfig(
            seq_name="toy_512_3",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_4 = TrainConfig(
        work_dir="./outdir",
        data=CustomDataConfig(
            seq_name="toy_512_4",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    main([config_1, config_2, config_3, config_4])
    #main(config_1, config_2, config_3, config_4)
