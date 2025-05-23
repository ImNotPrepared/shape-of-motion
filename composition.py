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
def mask_overlap_ratio(masks1, masks2):
    """
    Compute Overlapping Ratio between each pair of masks in masks1 and masks2.
    
    Args:
        masks1 (torch.Tensor): Tensor of shape [N, H, W], binary masks.
        masks2 (torch.Tensor): Tensor of shape [M, H, W], binary masks.

    Returns:
        overlap_ratio (torch.Tensor): Overlap ratio matrix of shape [N, M] where each entry is
                                      (intersection area / area of masks1[i]).
    """
    masks1 = masks1.bool()
    masks2 = masks2.bool()
    N, H, W = masks1.shape
    M = masks2.shape[0]

    # Flatten masks for easier computation
    masks1 = masks1.view(N, -1)  # [N, H*W]
    masks2 = masks2.view(M, -1)  # [M, H*W]

    # Compute intersection
    intersection = (masks1.unsqueeze(1) & masks2.unsqueeze(0)).sum(dim=-1).float()  # [N, M]

    # Compute the area of each mask in masks1
    area_masks1 = masks1.sum(dim=-1, keepdim=True).float()  # [N, 1]

    # Avoid division by zero
    overlap_ratio = intersection / (area_masks1 + 1e-6)  # [N, M]
    return overlap_ratio

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
        print(len(train_dataset))
        ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            num_workers=cfg.num_dl_workers,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn,
        )
        
        train_indices = cfg.train_indices


        scores = []
        for first_batch in train_loader:
          first_batch = first_batch['masks']
          print(first_batch.shape)
          for index in range(len(first_batch)-1):
            scores.append(mask_overlap_ratio(first_batch[index][None, ...], first_batch[index+1][None, ...]))
          print(min(scores))
          
        # val_img_datases

        # train_list.append((train_video_view, train_loader, train_dataset, train_video_view))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = [train[1] for train in train_list]
    train_loaders = [train_loader[i] for i in train_indices]


    train_video_views = [train[0] for train in train_list]


    train_dataset = [train[2] for train in train_list]
    train_datasets = [train_dataset[i] for i in train_indices]
    print('wtf'*50, len(train_dataset), len(train_datasets))



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

    import json
    # /data3/zihanwa3/Capstone-DSR/Processing_panoptic_tennis
    try: 
        md = json.load(open(cfg.test_w2cs, 'r'))
    except:
        md = json.load(open(cfg.test_w2cs.replace('raw_data/', 'Processing_'), 'r'))
        
    c2ws = []
    for c in range(1, 6):
      if c==5:
          c=1
      k, w2c =  md['k'][0][c], np.linalg.inv(md['w2c'][0][c])
      c2ws.append(w2c)


    all_interpolated_c2ws = []
    for i in range(len(c2ws) - 1):
        c2w1 = c2ws[i]
        c2w2 = c2ws[i + 1]
        interpolated = interpolate_cameras(c2w1, c2w2, alpha=0.5)
        all_interpolated_c2ws.extend(interpolated)

    all_interpolated_c2ws = []

    t = 0
    ### 1, 3 to 22 
    ## NECESSARY: 0 - 》 1 3 8 23  [1, 3, 8, 23]
    ### replaceable: 3/13  21/23 choce one 
    for c in [8, 9, 10, 11]:
        all_interpolated_c2ws.append((md['w2c'][t][c]))
    all_interpolated_c2ws = np.array(all_interpolated_c2ws)
    
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

        # Zip the loaders to load one batch from each loader at each step
        #for batches in syn_dataloader:
        #    batches = to_device(batches, device)

        for batches in zip(*train_loaders):
            batches = [to_device(batch, device) for batch in batches]
            loss = trainer.train_step(batches)
            loss.backward()
            trainer.op_af_bk()

            pbar.set_description(f"Loss: {loss:.6f}")

            
        if (epoch > 0 and epoch % cfgs[0].save_videos_every == 0) or (
            epoch == cfgs[0].num_epochs - 1
        ):
            for iiidx, validator in enumerate(validators):
                validator.save_int_videos(epoch, all_interpolated_c2ws[iiidx])
                validator.save_train_videos_images(epoch)



        if (epoch > 0 and epoch % cfg.validate_every == 0) or (
            epoch == cfg.num_epochs - 1
        ):
            for ind, validator in enumerate(validators):
              val_logs = validator.validate()
              metrics_str = "\n".join([f"{key}: {value}" for key, value in val_logs.items()])

              with open(f"{cfg.work_dir}/validation_metrics_cam{ind}.txt", "a") as f:  
                  f.write(f"Epoch {epoch}\n")
                  f.write(metrics_str + "\n\n")

    #####
    #
    #
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
    debug=False,
    seq_name: str = ''
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
        seq_name=seq_name
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
    seq_name: str = ''
):
    # Prepare lists to collect data from each dataset
    tracks_3d_list = []
    visibles_list = []
    invisibles_list = []
    confidences_list = []
    colors_list = []
    feats_list = []
    tracks_3d = None
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

    #init_fg_motion_bases_from_single_t()
    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )

    motion_bases = motion_bases.to(device)
    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    old_method = True
    if seq_name != 'dance':
      if old_method == True:
        motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
            tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
        )

        motion_bases = motion_bases.to(device)
        fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
      else:
        cano_t = 216 # int((1615-1477) / 3)
        # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp_newgraph/1615/fg_proj_img_0.png
        #if seq_name == 'dance':
        #    cano_t = #/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp/1528
        org_path=f'/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512_Aligned_fg_only/'
        fg_depth_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_bike_100_bs1/'
        #'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp/'
        from flow3d.org_utils import get_preset_data, get_preset_dance
        get_data = get_preset_data(512)

        motion_bases, motion_coefs, fg_params = init_fg_motion_bases_from_single_t(tracks_3d, num_motion_bases, rot_type, cano_t,                                                             
            get_data=get_data, org_path=org_path, fg_depth_path=fg_depth_path, )#, sampled_centers
      ##### OUTPUT: MotionBases, fg_params
    ## CAN BE REPLACED BY:
    # init_fg_motion_bases_from_single_t

    motion_coefs = motion_coefs#.float()
    fg_params = fg_params.to(device)

    bg_params = None
    if num_bg > 0:
        bg_points_list = []
        bg_normals_list = []
        bg_colors_list = []
        bg_feats_list = []

        for item, train_dataset in enumerate(train_datasets):
            
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
        bg_params = init_bg(bg_points, seq_name=seq_name)
        bg_params = bg_params.to(device)

    
    if tracks_3d:
      tracks_3d = tracks_3d.to(device)
    return fg_params, motion_bases, bg_params, tracks_3d

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