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
    num_fg: int = 100_000
    num_bg: int = 0 ### changed to 0
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
      train_loader = DataLoader(
          train_dataset,
          batch_size=cfg.batch_size,
          num_workers=cfg.num_dl_workers,
          persistent_workers=True,
          collate_fn=BaseDataset.train_collate_fn,
      )

      train_list.append((train_loader , train_loader, train_dataset))



    train_dataset_0 = train_list[0][-1]#.train_step(batch_0)
    train_dataset_1 = train_list[1][-1]#.train_step(batch_1)
    train_dataset_2 = train_list[2][-1]#.train_step(batch_2)
    train_dataset_3 = train_list[3][-1]#.train_step(batch_3)

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
          [train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3, ],
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

            
            if debug:
              loss = trainer.train_step([batch_0, batch_0, batch_0, batch_0])

            else:
              loss = trainer.train_step([batch_0, batch_1, batch_2, batch_3])#(loss_0 + loss_1 + loss_2 + loss_3) / 4
            loss.backward()
            trainer.op_af_bk()

            pbar.set_description(f"Loss: {loss:.6f}")


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


        run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs, num_iters=1000)
        #print(fg_params.shape, motion_bases.shape)#, tracks_3d, Ks, w2cs)
        Ks_fuse.append(Ks)
        w2cs_fuse.append(w2cs)
        fg_params_fuse.append(fg_params)
        motion_bases_fuse.append(motion_bases)

        bg_params_fuse.append(bg_params)
        

    Ks_fuse = torch.cat(Ks_fuse, dim=0)  # Flatten [N, Ks] to [N * Ks]
    w2cs_fuse = torch.cat(w2cs_fuse, dim=0)  # Flatten w2cs similarly


    prefix="params."

    fg_state_dict_fused = {} 

    for key in fg_params_fuse[0].params.keys():
        print(key)
        fg_state_dict_fused[prefix+key] = torch.cat(
            [fg_params.params[key] for fg_params in fg_params_fuse], dim=0
        )

    nummms0 = len(fg_params_fuse[0].params['motion_coefs'])
    nummms1 = len(fg_params_fuse[1].params['motion_coefs'])
    nummms2 = len(fg_params_fuse[2].params['motion_coefs'])
    nummms3 = len(fg_params_fuse[3].params['motion_coefs'])

    base_nummms0 = fg_params_fuse[0].params['motion_coefs'].shape[1]
    base_nummms1 = fg_params_fuse[1].params['motion_coefs'].shape[1]
    base_nummms2 = fg_params_fuse[2].params['motion_coefs'].shape[1]
    base_nummms3 = fg_params_fuse[3].params['motion_coefs'].shape[1]
    to_init = torch.zeros((len(fg_state_dict_fused[prefix+'motion_coefs']), base_nummms0+base_nummms1+base_nummms2+base_nummms3))


    if debug:
      to_init[:nummms0, :10] = fg_params_fuse[0].params['motion_coefs']
      to_init[nummms0:nummms1+nummms0, 10:20] = fg_params_fuse[1].params['motion_coefs']
      to_init[nummms1+nummms0:nummms2+nummms1+nummms0, 20:30] = fg_params_fuse[2].params['motion_coefs']
      to_init[nummms2+nummms1+nummms0:, 30:] = fg_params_fuse[3].params['motion_coefs']

    else: 
      to_init[:nummms0, :base_nummms0] = fg_params_fuse[0].params['motion_coefs']
      to_init[nummms0:nummms1+nummms0, base_nummms0:base_nummms0+base_nummms1] = fg_params_fuse[1].params['motion_coefs']
      to_init[nummms1+nummms0:nummms2+nummms1+nummms0, base_nummms0+base_nummms1:base_nummms0+base_nummms1+base_nummms2] = fg_params_fuse[2].params['motion_coefs']
      to_init[nummms2+nummms1+nummms0:, base_nummms0+base_nummms1+base_nummms2:] = fg_params_fuse[3].params['motion_coefs']


    fg_state_dict_fused[prefix+'motion_coefs'] = to_init

    bg_state_dict_fused = {} 

    try:
      for key in bg_params_fuse[0].params.keys():
          bg_state_dict_fused[prefix+key] = torch.cat(
              [bg_params.params[key] for bg_params in bg_params_fuse], dim=0
          )
    except:
      print('NO_BG')

    #fg_params_fuse[0].scene_center
    #for key in ['scene_center', 'scene_scale']:
    #    fg_state_dict_fused[key] = torch.cat(
    #        [fg_params[key] for fg_params in fg_params_fuse], dim=0
    #    )
        
    motion_bases_state_dict_fused = {}
    for key in motion_bases_fuse[0].params.keys():
        motion_bases_state_dict_fused[prefix+key] = torch.cat([d.params[key] for d in motion_bases_fuse], dim=0)

    fg_params_fused = GaussianParams.init_from_state_dict(fg_state_dict_fused)
    motion_bases_fused = MotionBases.init_from_state_dict(motion_bases_state_dict_fused)
    bg_params_fused = GaussianParams.init_from_state_dict(bg_state_dict_fused)


    #fg_params_fuse = torch.cat(fg_params_fuse, dim=0)  # Flatten fg_params
    #motion_bases_fuse = torch.cat(motion_bases_fuse, dim=0)  # Flatten motion_bases
    #bg_params_fuse = torch.cat(bg_params_fuse, dim=0)  # Flatten bg_params

    model = SceneModel(Ks_fuse, w2cs_fuse, fg_params_fused, motion_bases_fused, bg_params_fused)
    print('SANITY_CCCCHECK', model.fg.get_coefs().shape, model.fg.get_colors().shape)

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
    import wandb 
    import argparse
    import wandb
    import tyro

    wandb.init()  

    work_dir = './output_duster_feature_rendering'
    config_1 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="toy_512_1",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_2 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="toy_512_2",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_3 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="toy_512_3",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    config_4 = TrainConfig(
        work_dir=work_dir,
        data=CustomDataConfig(
            seq_name="toy_512_4",
            root_dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/data",
        ),
        lr=tyro.cli(SceneLRConfig),
        loss=tyro.cli(LossesConfig),
        optim=tyro.cli(OptimizerConfig),
    )
    main([config_1, config_2, config_3, config_4])