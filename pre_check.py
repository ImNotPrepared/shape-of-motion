import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated
import wandb
import numpy as np
import torch
import tyro
import yaml
from typing import Literal, cast
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm
from flow3d.vis.utils import (
    apply_depth_colormap,
    make_video_divisble,
    plot_correspondences,
)

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
import imageio as iio
from flow3d.params import GaussianParams, MotionBases
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server

torch.set_float32_matmul_precision("high")
# _wheel_only
new_mask_dir = "/data3/zihanwa3/Capstone-DSR/Processing_bike/sam_v2_dyn_mask"
gif_frames = []

video_dir=new_mask_dir
for sub_dir in ['1', '2', '3', '4']:
    masks = []
    new_mask_sub_dir = os.path.join(new_mask_dir, sub_dir)
    for file_name in sorted(os.listdir(new_mask_sub_dir)):
        if file_name.endswith('.npz'):
            new_mask_path = os.path.join(new_mask_sub_dir, file_name)
            combined_mask = np.load(new_mask_path)['dyn_mask'][0]
            mask = cast(torch.Tensor, combined_mask)
            mask=torch.tensor(mask)
            
            masks.append(mask)
    
    if len(masks) > 0:
        mask_video = torch.stack(masks, dim=0)
        iio.mimwrite(
            osp.join(video_dir, f"{sub_dir}_masks.mp4"),
            make_video_divisble((mask_video.numpy() * 255).astype(np.uint8)),
            fps=15,
        )

