import os
import time
from dataclasses import dataclass
import os
import time
from dataclasses import dataclass
import json
import torch
import tyro
import numpy as np
from loguru import logger as guru
import imageio
from flow3d.renderer import Renderer
import torch
import tyro
from loguru import logger as guru
import cv2
from flow3d.renderer import Renderer
from PIL import Image
import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated
from typing import cast
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


@dataclass
class RenderConfig:
    work_dir: str 


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
    num_bg: int = 100_000
    num_motion_bases: int = 10
    num_epochs: int = 500
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 8
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50
    
  
def main(cfg_1: RenderConfig, cfgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{cfg_1.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path)

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir=cfg_1.work_dir,
        port=None,
    )


    base_data_path = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego'
    seq='cmu_bike'

    file_path = os.path.join(base_data_path, seq, 'Dy_train_meta.json') 
    with open(file_path, 'r') as file:
        json_file = json.load(file)
    md=json_file
    base_visuals_path =f'{cfg_1.work_dir}/new_visuals'
    os.makedirs(base_visuals_path, exist_ok=True)


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


    train_loader_0 = train_list[0][1]
    train_loader_1 = train_list[1][1]
    train_loader_2 = train_list[2][1]
    train_loader_3 = train_list[3][1]

    for cam_idddds in [1, 2, 3 , 4]:
      images = []

      if cam_idddds == 1:
        train_loaders = [train_loader_0, train_loader_0, train_loader_0, train_loader_0]

      elif cam_idddds == 2:
        train_loaders = [train_loader_1, train_loader_1, train_loader_1, train_loader_1]


      elif cam_idddds == 3:
        train_loaders = [train_loader_2, train_loader_2, train_loader_2, train_loader_2]


      elif cam_idddds == 4:
        train_loaders = [train_loader_3, train_loader_3, train_loader_3, train_loader_3]
    
      for batch_0, batch_1, batch_2, batch_3 in zip(*train_loaders):
          rendered_all = []

          #### load multi-view data
          batch_0 = to_device(batch_0, device)
          batch_1 = to_device(batch_1, device)
          batch_2 = to_device(batch_2, device)
          batch_3 = to_device(batch_3, device)
          batch=[batch_0, batch_1, batch_2, batch_3]


          B = len(batch) * batch[0]["imgs"].shape[0]
          W, H = img_wh = batch[0]["imgs"].shape[2:0:-1]

          N = batch[0]["target_ts"][0].shape[0]

          ts = torch.cat([b["ts"] for b in batch], dim=0)  # (sum of B across batches,)
          #print(ts)
          # Concatenate world-to-camera matrices (B, 4, 4).
          w2cs = torch.cat([b["w2cs"] for b in batch], dim=0)  # (sum of B across batches, 4, 4)

          # Concatenate camera intrinsics (B, 3, 3).
          Ks = torch.cat([b["Ks"] for b in batch], dim=0)  # (sum of B across batches, 3, 3)

          # Concatenate images (B, H, W, 3).
          imgs = torch.cat([b["imgs"] for b in batch], dim=0)  # (sum of B across batches, H, W, 3)

          # Concatenate valid masks or create ones where masks are missing (B, H, W).
          valid_masks = torch.cat([b.get("valid_masks", torch.ones_like(b["imgs"][..., 0])) for b in batch], dim=0)  # (sum of B across batches, H, W)

          # Concatenate masks and apply valid_masks (B, H, W).
          masks = torch.cat([b["masks"] for b in batch], dim=0) * valid_masks  # (sum of B across batches, H, W)

          # Concatenate depth maps (B, H, W).
          depths = torch.cat([b["depths"] for b in batch], dim=0)  # (sum of B across batches, H, W)

          query_tracks_2d = [track for b in batch for track in b["query_tracks_2d"]]
          target_ts = [ts for b in batch for ts in b["target_ts"]]


          loss = 0.0

          bg_colors = []
          rendered_all = []

          for i in range(B):
              rendered = renderer.model.render(
                  ts[i].item(),
                  w2cs[None, i],
                  Ks[None, i],
                  img_wh,
              )
              rendered_all.append(rendered)
          rendered_all = {
              key: (
                  torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                  if rendered_all[0][key] is not None
                  else None
              )
              for key in rendered_all[0]
          }
          rendered_imgs = cast(torch.Tensor, rendered_all["img"])
          #print(rendered_imgs.shape)

          for indiceees in range(len(rendered_imgs)):
            rendered_img = np.array(rendered_imgs[indiceees].detach().cpu().numpy()) * 255
            images.append(np.array(Image.fromarray((rendered_img).astype(np.uint8))))
      print(f'saving to {(base_visuals_path)}')
      imageio.mimsave(os.path.join(base_visuals_path, f'cam_{cam_idddds}.gif'), images, fps=5)


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


    render_fig = RenderConfig(work_dir='outdir_duster_depth')
    main(render_fig, [config_1, config_2, config_3, config_4])
    
    # main(tyro.cli(RenderConfig), [config_1, config_2, config_3, config_4])
