import os
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast
from torch.utils.data import Dataset
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from roma import roma
from tqdm import tqdm
import glob
from flow3d.data.base_dataset import BaseDataset
from flow3d.data.utils import (
    UINT16_MAX,
    SceneNormDict,
    get_tracks_3d_for_query_frame,
    median_filter_2d,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import json 
@dataclass
class DavisDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = "480p"
    image_type: str = "JPEGImages"
    mask_type: str = "Annotations"
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 3
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False


@dataclass
class CustomDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = ""
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: str = "modest"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 1
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    video_name: str = ''
    super_fast: bool = False



class CasualDataset(BaseDataset):
    def __init__(
        self,
        seq_name: str,
        root_dir: str,
        start: int = 0,
        end: int = -1,
        res: str = "480p",
        image_type: str = "JPEGImages",
        mask_type: str = "Annotations",
        depth_type: str = "aligned_depth_anything",
        camera_type: Literal["droid_recon"] = "droid_recon",
        track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir",
        mask_erosion_radius: int = 3,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        video_name: str = '_bike',
        super_fast: bool=False,
        **_,
    ):
        super().__init__()
        #/data3/zihanwa3/Capstone-DSR/Appendix/Depth-Anything-V2/new_scales_shifts.json
        pathhh = '/data3/zihanwa3/Capstone-DSR/Appendix/Depth-Anything-V2/new_scales_shifts.json'
        with open(pathhh, 'r') as f:
            scales_shifts = json.load(f)['scales_shifts']
        self.scales_shifts=scales_shifts
        self.seq_name = seq_name
        self.root_dir = root_dir
        self.res = res
        self.depth_type = depth_type
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False
        self.mask_erosion_radius = mask_erosion_radius

        self.img_dir = f"{root_dir}/{image_type}/{res}/{seq_name}"

        self.img_ext = os.path.splitext(os.listdir(self.img_dir)[0])[1]

        self.feat_dir = f"{root_dir}/{image_type}/{res}/{seq_name}"
        self.feat_ext = os.path.splitext(os.listdir(self.feat_dir)[0])[1]

        # category = {video_name.split("_")[2]}
        path = f'/data3/zihanwa3/Capstone-DSR/Processing{video_name}/undist_cam01/*.jpg'
        paths = glob.glob(path)
        sorted_paths = sorted(paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
        try:
          self.min_ = int(os.path.basename(sorted_paths[0]).split('.')[0])
          self.max_= int(os.path.basename(sorted_paths[-1]).split('.')[0])
        except:
          self.min_, self.max_ = 0, 150


        # self.camera_path
        self.video_name = video_name# '_dance'
        #print(self.video_name, 'self.video name')
        self.hard_indx_dict = {
          '_bike': [49, 349, 3], 
          '_dance': [1477, 1778, 3],
          '': [273, 294, 1],
        }
        self.glb_first_indx = self.min_ #self.hard_indx_dict[self.video_name][0]
        self.glb_last_indx = self.max_ #self.hard_indx_dict[self.video_name][1]
        self.glb_step = 3 #self.hard_indx_dict[self.video_name][2]

        self.depth_dir = f"{root_dir}/aligned_depth_anything/{res}/{seq_name}"
        self.mask_dir = f"{root_dir}/{mask_type}/{res}/{seq_name}"
        self.tracks_dir = f"{root_dir}/{track_2d_type}/{res}/{seq_name}"
        self.cache_dir = f"{root_dir}/flow3d_preprocessed/{res}/{seq_name}"

        frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]
        if super_fast:
          frame_names=frame_names[-22:]
        #print(frame_names)
        #print(self.video_name)

        if end == -1:
            end = len(frame_names)
        self.start = start
        self.end = end
        if self.video_name=='_bike':
          self.frame_names = frame_names[start:end:self.glb_step][:-1]
        elif self.video_name=='_dance':
          self.frame_names = frame_names[start:end:self.glb_step]#[:-1]
        elif self.video_name=='':
          self.frame_names = frame_names[start:end:self.glb_step][:-1]
        else:
          self.frame_names = frame_names[start:end:self.glb_step]

        frame_names=self.frame_names
        # print(self.start, self.end)

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.feats: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]



        self.debug=False

        def load_known_cameras_panoptic(
            path: str, H: int, W: int, noise: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
          import json
          import numpy as np
          size = W
          
          seq = seq_name.split('_')[0]
          json_path = f'/data3/zihanwa3/Capstone-DSR/monst3r_train/my_data_2/Dynamic3DGaussians/data/{seq}/train_meta.json'

          with open(json_path, 'r') as f:
              df = json.load(f)
          # 0 10 15 30
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

          intrinsics = np.array([df['k'][0][i] for i in cam_ids])
          w2cs = np.array([df['w2c'][0][i] for i in cam_ids])
          # print(np.array(intrinsics).shape, np.array(w2cs).shape)

          def matrix_to_pose(extrinsic):
              """
              Convert a 4x4 extrinsic matrix to [tx, ty, tz, qw, qx, qy, qz].
              """
              # Extract rotation (3×3) and translation (3×1)
              rotation_matrix = extrinsic[:3, :3]
              translation = extrinsic[:3, 3]
              
              # Convert rotation matrix to quaternion [x, y, z, w]
              quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
              # By default, scipy returns [x, y, z, w]
              qx, qy, qz, qw = quat_xyzw
              
              # Return in format [tx, ty, tz, qw, qx, qy, qz]
              return [translation[0], translation[1], translation[2], 
                      qw, qx, qy, qz]

          # Example usage:
          # Suppose w2cs is an ndarray of shape (N, 4, 4)
          # w2cs[i] = the 4x4 extrinsic for camera i
          # np.linalg.inv

          # poses = [matrix_to_pose((w2cs[i])) for i in range(len(w2cs))]

          def k_to_intrinsics(k_matrix, image_width, image_height):
              """
              Convert a 3x3 K matrix to [image_width, image_height, fx, fy, cx, cy].
              """
              fx = k_matrix[0, 0]
              fy = k_matrix[1, 1]
              cx = k_matrix[0, 2]
              cy = k_matrix[1, 2]
              
              return [image_width, image_height, fx, fy, cx, cy]

          intrinsics_list = [k_to_intrinsics(intrinsics[i], 512, 288) 
                            for i in range(len(intrinsics))]
          #poses = df[['tx_world_cam', 'ty_world_cam', 'tz_world_cam', 'qw_world_cam', 'qx_world_cam', 'qy_world_cam', 'qz_world_cam',  ]].values.tolist()
              # pose: [tx, ty, tz, qw, qx, qy, qz]
          # 3.  [1.457692, -0.240018, -0.077916, -0.522571, -0.55499, 0.436684, 0.477716],
          # intrinsics = df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist()
          #       3.         [1764.426025, 1764.426025, 1920.0, 1080.0],

          poses=(w2cs[:]) # 
          intrinsics= intrinsics_list[:]

          def convert_to_matrix(pose):
              tx, ty, tz, qw, qx, qy, qz = pose
              rotation = R.from_quat([qx, qy, qz, qw])
              rotation_matrix = rotation.as_matrix()
              #(x, y, z, w)
              
              transformation_matrix = np.eye(4)
              transformation_matrix[:3, :3] = rotation_matrix
              transformation_matrix[:3, 3] = [tx, ty, tz]
              
              return transformation_matrix
          pose_matrices = poses# [(convert_to_matrix(pose)) for pose in poses]
          # print(w2cs.shape)

          def convert_intrinsics_to_matrix(intrinsics, size):
              _, _, fx, fy, cx, cy = intrinsics
              ratio = size/640
              fx *= ratio
              fy *= ratio
              cx *= ratio
              cy *= ratio
              intrinsics_matrix = np.array([
                  [fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]
              ])
              
              return intrinsics_matrix

          intrinsics_matrices = [convert_intrinsics_to_matrix(intrinsics, size) for intrinsics in intrinsics]

          pose_matrices = np.array(pose_matrices)
          intrinsics_matrices = np.array(intrinsics_matrices)

          cam_iddddds = {
            '3': 0,
            '21': 1, 
            '23': 2,
            '25': 3,
          }
          c = int(self.seq_name[-2:])
          posessss, intrinsicsss = [], []
          for t in range(self.glb_first_indx, self.glb_last_indx, self.glb_step):
            pose_matrice, intrinsics_matrice = pose_matrices[cam_iddddds[str(c)]], intrinsics_matrices[cam_iddddds[str(c)]]
            posessss.append(pose_matrice)
            intrinsicsss.append(intrinsics_matrice)


          pose_matrices = torch.tensor(posessss).float()
          intrinsics_matrices = torch.tensor(intrinsicsss).float()


        
          return pose_matrices, intrinsics_matrices, None 

        def load_known_cameras(
            path: str, H: int, W: int, noise: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert os.path.exists(path), f"Camera file {path} does not exist."
            md = json.load(open(path, 'r'))
            c2ws = []
            #for c in range(4, 5):
            c = int(self.seq_name[-1])
            for t in range(self.glb_first_indx, self.glb_last_indx, self.glb_step):
              h, w = md['hw'][c]
              k, w2c =  md['k'][t][c], np.linalg.inv(md['w2c'][t][c])
              if noise:
                R = w2c[:3, :3]
                t = w2c[:3, 3]

                # Define the maximum deviation in degrees and convert to radians
                max_deviation_deg = 5
                max_deviation_rad = np.deg2rad(max_deviation_deg)

                # Generate random rotation angles within ±5 degrees for each axis
                noise_angles = np.random.uniform(-max_deviation_rad, max_deviation_rad, size=3)

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

              c2ws.append(w2c[None, ...])

            traj_c2w = np.concatenate(c2ws)
            sy, sx = H / h, W / w
            fx, fy, cx, cy = k[0][0],  k[1][1], k[0][2], k[1][2], # (4,)

            K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])  # (3, 3)
            Ks = np.tile(K[None, ...], (len(traj_c2w), 1, 1))  # (N, 3, 3)

            #path='/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/droid_recon/toy_4.npy'
            #recon = np.load(path, allow_pickle=True).item()
            #kf_tstamps = recon["tstamps"].astype("int")
            #kf_tstamps = torch.from_numpy(kf_tstamps)
            kf_tstamps=None
            return (
                torch.from_numpy(traj_c2w).float(),
                torch.from_numpy(Ks).float(),
                kf_tstamps,
            )
        

        if camera_type == "droid_recon":
            img = self.get_image(0)
            H, W = img.shape[:2]
            # path = "/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/Dy_train_meta.json"
            path = f'/data3/zihanwa3/Capstone-DSR/raw_data/{self.video_name[1:]}/trajectory/Dy_train_meta.json'
            if self.debug:
              w2cs, Ks, tstamps = load_cameras(
                  f"{root_dir}/{camera_type}/{seq_name}.npy", H, W
              )
            else:
              try:
                w2cs, Ks, tstamps = load_known_cameras(
                path, H, W, noise=False ##############################FUKKKING DOG
                )
              except:
                w2cs, Ks, tstamps = load_known_cameras_panoptic(
                path, H, W, noise=False ##############################FUKKKING DOG
                )          
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        assert (
            len(frame_names) == len(w2cs) == len(Ks)
        ), f"{len(frame_names)}, {len(w2cs)}, {len(Ks)}"


        self.w2cs = w2cs[start:end]
        self.Ks = Ks[start:end]
        if tstamps is None:
          self._keyframe_idcs=None
        else:
          tmask = (tstamps >= start) & (tstamps < end)
          self._keyframe_idcs = tstamps[tmask] - start
          
        self.scale = 1

        if scene_norm_dict is None:
            cached_scene_norm_dict_path = os.path.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if os.path.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                guru.info("loading cached scene norm dict...")
                scene_norm_dict = torch.load(
                    os.path.join(self.cache_dir, "scene_norm_dict.pth")
                )
            else:
                tracks_3d = self.get_tracks_3d(5000, step=self.num_frames // 10)[0]
                #scale, transfm = compute_scene_norm(tracks_3d, self.w2cs)
                #scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
                #os.makedirs(self.cache_dir, exist_ok=True)
                #torch.save(scene_norm_dict, cached_scene_norm_dict_path)

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return self._keyframe_idcs

    def __len__(self):
        return len(self.frame_names)

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks

    def get_images(self):
        imgs = [cast(torch.Tensor, self.load_image(index)) for index in range(len(self.frame_names))]
        return imgs

    def get_img_wh(self) -> tuple[int, int]:
        return self.get_image(0).shape[1::-1]

    def get_bkgd_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = False,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        end = self.num_frames
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )

        ###  start 
        #if use_kf_tstamps:
        #    query_idcs = self.keyframe_idcs.tolist()
        #else:
        num_query_frames = self.num_frames // stride
        query_endpts = torch.linspace(start, end, num_query_frames + 1)
        query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        print(f"{query_idcs=}")
        for query_idx in tqdm(query_idcs, desc="Loading bkgd points", leave=False):
            img = self.get_image(query_idx)
            feat = self.get_feat(query_idx)
            depth = self.get_depth(query_idx)
            bg_mask = self.get_mask(query_idx) < 0
            bool_mask = (bg_mask * (depth > 0)).to(torch.bool)

            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]

            # get the bounding box of previous points that reproject into frame
            # inefficient but works for now
            bmax_x, bmax_y, bmin_x, bmin_y = 0, 0, W, H
            for p3d, _, _, _ in bg_geometry:
                if len(p3d) < 1:
                    continue
                # reproject into current frame
                p2d = torch.einsum(
                    "ij,jk,pk->pi", K, w2c[:3], F.pad(p3d, (0, 1), value=1.0)
                )
                p2d = p2d[:, :2] / p2d[:, 2:].clamp(min=1e-6)
                xmin, xmax = p2d[:, 0].min().item(), p2d[:, 0].max().item()
                ymin, ymax = p2d[:, 1].min().item(), p2d[:, 1].max().item()

                bmin_x = min(bmin_x, int(xmin))
                bmin_y = min(bmin_y, int(ymin))
                bmax_x = max(bmax_x, int(xmax))
                bmax_y = max(bmax_y, int(ymax))

            # don't include points that are covered by previous points
            bmin_x = max(0, bmin_x)
            bmin_y = max(0, bmin_y)
            bmax_x = min(W, bmax_x)
            bmax_y = min(H, bmax_y)
            overlap_mask = torch.ones_like(bool_mask)
            overlap_mask[bmin_y:bmax_y, bmin_x:bmax_x] = 0

            bool_mask &= overlap_mask
            if bool_mask.sum() < min_per_frame:
                guru.debug(f"skipping {query_idx=}")
                continue

            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]
            point_feats = feat[bool_mask]

            num_sel = max(len(points) // down_rate, min_per_frame)
            sel_idcs = np.random.choice(len(points), num_sel, replace=False)
            points = points[sel_idcs]
            point_normals = point_normals[sel_idcs]
            point_colors = point_colors[sel_idcs]
            point_feats = point_feats[sel_idcs]
            guru.debug(f"{query_idx=} {points.shape=}")
            bg_geometry.append((points, point_normals, point_colors, point_feats))

        bg_points, bg_normals, bg_colors, bg_feats = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )
        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]
            bg_feats = bg_feats[sel_idcs]

        return bg_points, bg_normals, bg_colors, bg_feats

    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, step))

        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
        fg_masks = (masks == 1).float()
        depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
        inv_Ks = torch.linalg.inv(self.Ks[target_idcs])
        c2ws = torch.linalg.inv(self.w2cs[target_idcs])

        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs)))
        cur_num = 0
        tracks_all_queries = []
        for q_idx in query_idcs:
            # (N, T, 4)
            tracks_2d = self.load_target_tracks(q_idx, target_idcs)
            num_sel = int(
                min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
            )
            if num_sel < len(tracks_2d):
                sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                tracks_2d = tracks_2d[sel_idcs]
            cur_num += tracks_2d.shape[0]
            img = self.get_image(q_idx)
            feat = self.get_feat(q_idx)


            tidx = target_idcs.index(q_idx)


            tracks_tuple = get_tracks_3d_for_query_frame(
                tidx, img, tracks_2d, depths, fg_masks, inv_Ks, c2ws, feat
            )
  
            tracks_all_queries.append(tracks_tuple)


        tracks_3d, colors, feats, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )


        return tracks_3d, visibles, invisibles, confidences, colors, feats

    def get_image(self, index) -> torch.Tensor:
        if self.imgs[index] is None:
            self.imgs[index] = self.load_image(index)
        img = cast(torch.Tensor, self.imgs[index])
        return img

    def get_feat(self, index) -> torch.Tensor:
        if self.feats[index] is None:
            self.feats[index] = self.load_feat(index)
        feat = cast(torch.Tensor, self.feats[index])
        return feat

    def get_mask(self, index) -> torch.Tensor:
        if self.masks[index] is None:
            self.masks[index] = self.load_mask(index)
        
        mask = cast(torch.Tensor, self.masks[index])
        return mask

    def get_masks(self):
        masks = [cast(torch.Tensor, self.load_mask(index)) for index in range(len(self.frame_names))]
        return masks

    def get_depth(self, index) -> torch.Tensor:
        if self.depths[index] is None:
            self.depths[index] = self.load_depth(index)
        return self.depths[index] #/ self.scale

    def get_depths(self):
        depths = [cast(torch.Tensor, self.load_depth(index)) for index in range(len(self.frame_names))]
        return depths

    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0

    def load_feat(self, index) -> torch.Tensor:
        # path = f"{self.feat_dir}/{self.frame_names[index]}{self.feat_ext}"¸¸
        path = f"{self.feat_dir}/{self.frame_names[index]}{self.feat_ext}"
        path = path.replace('toy_512_', 'undist_cam0') # cam0x              ############# _fg_only                                                              _fg_only/
        path = path.replace('jpg', 'npy')
        ### examples:
        # /data3/zihanwa3/Capstone-DSR/Processing_dance/dinov2features/resized_512_Aligned_fg_only/undist_cam01 

        #try:
        #  dcpath = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/dinov2features/resized_512_Aligned/')
        #  dinov2_feature = torch.tensor(np.load(dcpath)).to(torch.float32)
        #except:
        ddpath = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images//', 
        f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/dinov2features/resized_512_Aligned_fg_only_is_True/')
        # fg_only_is_False
        
        dinov2_feature = torch.tensor(np.load(ddpath)).to(torch.float32)          
        return dinov2_feature

    def load_mask(self, index) -> torch.Tensor:

        if self.debug:
          path = f"{self.mask_dir}/{self.frame_names[index]}.png"
          r = self.mask_erosion_radius
          mask = imageio.imread(path)

        else:
          try:
            path = f"{self.mask_dir}/{self.frame_names[index]}.npz"
            r = self.mask_erosion_radius
            mask = np.load(path)['dyn_mask'][0][:, :, None].repeat(3, axis=2)
          except:
            try:
              path = f"{self.mask_dir}/dyn_mask_{int(self.frame_names[index])}.npz"
              r = self.mask_erosion_radius
              mask = np.load(path)['dyn_mask'][0][:, :, None].repeat(3, axis=2)
            except:
              #/data3/zihanwa3/Capstone-DSR/Processing/sam_v2_dyn_mask/3
              new_mask_dir = f'/data3/zihanwa3/Capstone-DSR/Processing/sam_v2_dyn_mask/{self.mask_dir[-1]}'
              #print(self.mask_dir)
              path = f"{new_mask_dir}/dyn_mask_{int(self.frame_names[index])}.npz"
              mask = np.load(path)['dyn_mask'][0][:, :, None].repeat(3, axis=2)
              
              
                   
        # 2160, 3840, 1

        fg_mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        bg_mask = ~fg_mask
        fg_mask_erode = cv2.erode(
            fg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        bg_mask_erode = cv2.erode(
            bg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        out_mask = np.zeros_like(fg_mask, dtype=np.float32)
        out_mask[bg_mask_erode > 0] = -1
        out_mask[fg_mask_erode > 0] = 1
        return torch.from_numpy(out_mask).float()

    def load_org_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        disp = np.load(path)
        depth = 1.0 / np.clip(disp, a_min=1e-6, a_max=1e6)
        depth = torch.from_numpy(depth).float()
        depth = median_filter_2d(depth[None, None], 11, 1)[0, 0]
        return depth

    def load_target_tracks(
        self, query_index: int, target_indices: list[int], dim: int = 1
    ):
        """
        tracks are 2d, occs and uncertainties
        :param dim (int), default 1: dimension to stack the time axis
        return (N, T, 4) if dim=1, (T, N, 4) if dim=0
        """
        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            path = f"{self.tracks_dir}/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))

    def load_depth(self, index) -> torch.Tensor:
        #  load_da2_depth load_duster_depth load_org_depth
        if self.depth_type == 'modest':
           depth = self.load_modest_depth(index)
        elif self.depth_type == 'da2':
           depth = self.load_org_depth(index)
        elif self.depth_type == 'dust3r':
           depth = self.load_duster_depth(index)       
        elif self.depth_type == 'monst3r':
           depth = self.load_monster_depth(index)    
        elif self.depth_type == 'monst3r+dust3r':
           depth = self.load_duster_moncheck_depth(index)    
        elif self.depth_type == 'load_vanila_depth':
           depth = self.load_vanila_depth(index)
        elif self.depth_type == 'panoptic_gt':
           depth = self.load_panoptic_gt(index)
        return depth

    def load_panoptic_gt(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/conf_bg_depth_{int(self.frame_names[index])}.npy"
        #print(self.depth_dir, path, int(self.frame_names[index]))
        to_replace = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//'
        if self.video_name == '_dance':
          new_path =  f'/data3/zihanwa3/Capstone-DSR/monst3r/aligned_preset_k/'
          path = path.replace(to_replace, new_path)
          path = path.replace('disp', 'frame')
          path = path.replace('_undist_cam', '/undist_cam')
        elif self.video_name == '':
          new_path = f'/data3/zihanwa3/Capstone-DSR/monst3r/bike_aligned_preset_k/'
          path = path.replace('toy_512_', 'undist_cam')
          path = path.replace(to_replace, new_path)
          path = path.replace('disp', 'frame')
          path = path.replace('_undist_cam', '/undist_cam')

        else:
          seq = self.video_name.split('_')[-1]
          new_path = f'/data3/zihanwa3/Capstone-DSR/Processing_panoptic_{seq}/jono_depth/'
          
          path = path.replace('toy_512_', 'undist_cam')
          cheaty_way = path.split('/')[-2]
          path = path.replace(to_replace, new_path)
          cam_id = cheaty_way.split('cam')[-1]
          path = path.replace(f'softball_undist_cam03', '')
          #print(self.frame_names[index], 'frame_names')
          if int(cam_id) == 3:
            cam_id = 2
          else:
            cam_id = int(cam_id) - 3


          new_path  = os.path.join(new_path, str(int(self.frame_names[index])), f'conf_bg_depth_{cam_id}.npy')
          path = new_path

          #path = '/data3/zihanwa3/Capstone-DSR/Processing_panoptic_tennis/jono_depth'
          # duster_depth_clean_dance_512_4_mons_cp


          #/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/
          #aligned_depth_anything//softball_undist_cam03/conf_bg_depth_0.npy


        # /data3/zihanwa3/Capstone-DSR/Processing_panoptic_softball/
        # jono_depth/0/conf_bg_depth_0.npy ### t/cam

        depth_map = np.load(path)
        depth_map = np.clip(depth_map, a_min=1e-8, a_max=1e6)
        depth = torch.from_numpy(depth_map).float()
        input_tensor = depth.unsqueeze(0).unsqueeze(0) 

        # If you want to remove the added dimensions
        depth = input_tensor.squeeze(0).squeeze(0) 
        return depth

    def load_modest_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/disp_{int(self.frame_names[index])}.npy"
        #print(self.depth_dir, path, int(self.frame_names[index]))
        to_replace = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//'
        if self.video_name == '_dance':
          new_path =  f'/data3/zihanwa3/Capstone-DSR/monst3r/aligned_preset_k/'
        elif self.video_name == '':
          new_path = f'/data3/zihanwa3/Capstone-DSR/monst3r/bike_aligned_preset_k/'
          path = path.replace('toy_512_', 'undist_cam')
        else:
          new_path = f"/data3/zihanwa3/Capstone-DSR/monst3r/aligned_preset_k_"
          path = path.replace('toy_512_', 'undist_cam')
        path = ''
        # duster_depth_clean_dance_512_4_mons_cp
        path = path.replace(to_replace, new_path)
        path = path.replace('disp', 'frame')
        path = path.replace('_undist_cam', '/undist_cam')
        depth_map = np.load(path)
        depth_map = np.clip(depth_map, a_min=1e-8, a_max=1e6)
        depth = torch.from_numpy(depth_map).float()
        input_tensor = depth.unsqueeze(0).unsqueeze(0) 

        # If you want to remove the added dimensions
        depth = input_tensor.squeeze(0).squeeze(0) 
        return depth

    def load_monster_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/disp_{int(self.frame_names[index])}.npy"
        #print(self.depth_dir, path, int(self.frame_names[index]))
        to_replace = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//'
        if self.video_name == '_dance':
          new_path =  f'/data3/zihanwa3/Capstone-DSR/monst3r/aligned_preset_k/'
        else:#elif self.video_name == '':
          new_path = f'/data3/zihanwa3/Capstone-DSR/monst3r/bike_aligned_preset_k/'
          path = path.replace('toy_512_', 'undist_cam')
        # duster_depth_clean_dance_512_4_mons_cp
        path = path.replace(to_replace, new_path)
        path = path.replace('disp', 'frame')
        depth_map = np.load(path)
        depth_map = np.clip(depth_map, a_min=1e-8, a_max=1e6)
        depth = torch.from_numpy(depth_map).float()
        input_tensor = depth.unsqueeze(0).unsqueeze(0) 

        # If you want to remove the added dimensions
        depth = input_tensor.squeeze(0).squeeze(0) 
        return depth
    

    def load_modest_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/disp_{int(self.frame_names[index])}.npy"
        #print(self.depth_dir, path, int(self.frame_names[index]))
        to_replace = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//'
        if self.video_name == '_dance':
          new_path =  f'/data3/zihanwa3/Capstone-DSR/monst3r/aligned_preset_k/'
        elif self.video_name == '':
          new_path = f'/data3/zihanwa3/Capstone-DSR/monst3r/bike_aligned_preset_k/'
          path = path.replace('toy_512_', 'undist_cam')
        else:
          new_path = f"/data3/zihanwa3/Capstone-DSR/monst3r/aligned_preset_k_"
          path = path.replace('toy_512_', 'undist_cam')
        # duster_depth_clean_dance_512_4_mons_cp
        path = path.replace(to_replace, new_path)
        path = path.replace('disp', 'frame')
        path = path.replace('_undist_cam', '/undist_cam')
        depth_map = np.load(path)
        depth_map = np.clip(depth_map, a_min=1e-8, a_max=1e6)
        depth = torch.from_numpy(depth_map).float()
        input_tensor = depth.unsqueeze(0).unsqueeze(0) 

        # If you want to remove the added dimensions
        depth = input_tensor.squeeze(0).squeeze(0) 
        return depth

    def load_duster_moncheck_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        path = f"{self.depth_dir}/disp_{int(self.frame_names[index])}.npz"

        if 'dance' in self.video_name:
          path = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/duster_depth_new_2.7/')
          path = path.replace('toy_512_', '')
          path = path.replace('disp_', '')
          # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp_newgraph/1564/conf_depth_2.npy

          # print(path.split('/')[-1][:-4], path.split('/')[-2])
          # 1477 undist_cam01
          final_path = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp_newgraph/' + path.split('/')[-1][:-4] + '/' + 'conf_depth_' + str(int(path.split('/')[-2][-1])-1) + '.npy'
        elif 'bike' in self.video_name:
          path = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/duster_depth_new_2.7/')
          path = path.replace('toy_512_', '')
          path = path.replace('disp_', '')
          # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp_newgraph/1564/conf_depth_2.npy

          # print(path.split('/')[-1][:-4], path.split('/')[-2])
          # 1477 undist_cam01
          # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_bike_100_bs1/173/conf_depth_0.npy
          final_path = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_bike_100_bs1/' + path.split('/')[-1][:-4] + '/' + 'conf_depth_' + str(int(path.split('/')[-2][-1])-1) + '.npy'
        else:
          path = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/duster_depth_new_2.7/')
          path = path.replace('toy_512_', '')
          path = path.replace('disp_', '')
          # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp_newgraph/1564/conf_depth_2.npy

          # print(path.split('/')[-1][:-4], path.split('/')[-2])
          # 1477 undist_cam01
          # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_bike_100_bs1/173/conf_depth_0.npy
          depth_ttttype = 'conf_depth_' #'pc_depth_'
          final_path = f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/dumon_depth/' + path.split('/')[-1][:-4] + '/' + depth_ttttype + str(int(path.split('/')[-2][-1])-1) + '.npy'


                  
        disp_map =  np.load(final_path)
        depth_map = np.clip(disp_map, a_min=1e-8, a_max=1e6)
        depth = torch.from_numpy(depth_map).float()
        input_tensor = depth.unsqueeze(0).unsqueeze(0) 
        output_size = (288, 512)  # (height, width)
        resized_tensor = F.interpolate(input_tensor, size=output_size, mode='bilinear', align_corners=False)

        # If you want to remove the added dimensions
        depth = resized_tensor.squeeze(0).squeeze(0) 
        return depth
    
    def load_vanila_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        disp = np.load(path)
        depth = 1.0 / np.clip(disp, a_min=1e-6, a_max=1e6)
        depth = torch.from_numpy(depth).float()
        depth = median_filter_2d(depth[None, None], 11, 1)[0, 0]
        return depth
    
    def load_duster_depth(self, index) -> torch.Tensor:
# /data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything/
# /toy_512_4/00194.npy /data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//toy_512_4
# /data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/4/disp_0.npz
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        path = f"{self.depth_dir}/disp_{int(self.frame_names[index])}.npz"
        path = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/duster_depth_new_2.7/')
        path = path.replace('toy_512_', '')
        path = path.replace('disp_', '')
        try:
          final_path = f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/duster_depth_new_2.7/' + path.split('/')[-1][:-4] + '/' + path.split('/')[-2] + '.npz'
          disp_map =  np.load(final_path)['depth']
          depth_map = np.clip(disp_map, a_min=1e-8, a_max=1e6)
          depth = torch.from_numpy(depth_map).float()
          input_tensor = depth.unsqueeze(0).unsqueeze(0) 
        except:
          final_path = f"/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_300_testonly/{os.path.splitext(os.path.basename(path))[0]}/pc_depth_{os.path.basename(os.path.dirname(path))[-1]}.npy"

          disp_map =  np.load(final_path)#['depth']          
          depth_map = np.clip(disp_map, a_min=1e-8, a_max=1e6)
          depth = torch.from_numpy(depth_map).float()
          input_tensor = depth.unsqueeze(0).unsqueeze(0) 

        output_size = (288, 512)  # (height, width)
        resized_tensor = F.interpolate(input_tensor, size=output_size, mode='bilinear', align_corners=False)

        # If you want to remove the added dimensions
        depth = resized_tensor.squeeze(0).squeeze(0) 
        return depth

    def __getitem__(self, index: int, target_inds=None):
        #index = np.random.randint(0, self.num_frames)
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": torch.tensor(index),
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.get_image(index),
            "feats": self.get_feat(index),
             "depths": self.get_depth(index),
        }
        tri_mask = self.get_mask(index)
        valid_mask = tri_mask != 0  # not fg or bg
        mask = tri_mask == 1  # fg mask
        data["masks"] = mask.float()
        data["valid_masks"] = valid_mask.float()

        # (P, 2)
        query_tracks = self.load_target_tracks(index, [index])[:, 0, :2]
        if target_inds is None:
          target_inds = torch.from_numpy(
              np.random.choice(
                  self.num_frames, (self.num_targets_per_frame,), replace=False
              )
          )
        else:
          target_inds = torch.from_numpy(
              target_inds
          )
        # (N, P, 4)
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), dim=0)



        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[target_inds]
        data["target_Ks"] = self.Ks[target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]
        # (N, P).
        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
        # (N, H, W)
        target_depths = torch.stack([self.get_depth(i) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0]
        return data


class CasualDatasetVideoView(Dataset):
    """Return a dataset view of the video trajectory."""

    def __init__(self, dataset: CasualDataset):
        super().__init__()
        self.dataset = dataset
        self.fps = 15
        self.imgs = self.dataset.get_images()
        self.masks = self.dataset.get_masks()
        self.depths = self.dataset.get_depths()

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, index):

        return {
            "frame_names": self.dataset.frame_names[index],
            "ts": index,
            "w2cs": self.dataset.w2cs[index],
            "Ks": self.dataset.Ks[index],
            "imgs": self.imgs[index],
            "depths": self.depths[index],
            "masks": self.masks[index],
        }

class EgoDataset(Dataset):
    def __init__(self, t, md, seq, mode='stat_only', clean_img=True, depth_loss=False, debug_mode='no'):
        self.t = t + 1111
        self.md = md
        self.seq = seq
        self.mode = mode
        self.clean_img = clean_img
        self.depth_loss = depth_loss
        self.debug_mode = debug_mode

        self.dino_mask = True
        if self.dino_mask:
            self.directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR_49'
        else:
            self.directory = '/data3/zihanwa3/Capstone-DSR/Appendix/SR_7_pls'

        self.jpg_filenames = self.get_jpg_filenames(self.directory)
        self.near = 1e-7
        self.far = 70.0
        self.indices = []

        self.t = 0
        for lis in [self.jpg_filenames]:
            for iiiindex, c in sorted(enumerate(lis)):
                self.indices.append(('ego', iiiindex, c))

        # Initialize lists to hold preloaded data
        self.cams = []

        self.Ks=[]
        self.w2cs=[]
        self.images = []
        self.masks = []
        self.depths = []
        self.features = []
        self.ids = []
        self.antimasks = []
        self.visibilities = []
        print("Preloading data...")

        print(len(self.indices), 'leenen')
        for idx, (data_type, iiiindex, c) in enumerate(self.indices):
          cam = self.load_cam(c)
          im = self.load_im(c)
          mask_tensor = self.load_mask(c)
          anti_mask_tensor = mask_tensor < 1e-2

          self.cams.append(cam)
          self.Ks.append(torch.tensor(cam['K']))
          self.w2cs.append(torch.tensor(cam['w2c']))

          self.images.append(im)
          self.masks.append(mask_tensor)
          self.antimasks.append(anti_mask_tensor)
          self.ids.append(iiiindex)
          self.visibilities.append(True)

          #depth = self.load_depth(c)
          feature = self.load_feature(c)
          #self.depths.append(depth)
          self.features.append(feature)

        print("Data preloading complete.")

    def get_jpg_filenames(self, directory):
        jpg_files = [int(file.split('.')[0]) for file in os.listdir(directory) if file.endswith('.jpg')]
        return jpg_files

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = {}
        cam = self.cams[idx]
        w2cs = self.w2cs[idx].float()
        K = self.Ks[idx].float()

        im = self.images[idx]
        id_ = self.ids[idx]
        data['w2cs'] = w2cs.float()
        data['Ks'] = K.float()
        data['imgs'] = im.float()
        data['id'] = id_
        data['valid_masks'] = self.antimasks[idx].float()
        data['depths'] = self.depths[idx]
        data['feats'] = self.features[idx].float()

        return data
    def get_bkgd_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = False,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        end = self.num_frames
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )

        ###  start 
        #if use_kf_tstamps:
        #    query_idcs = self.keyframe_idcs.tolist()
        #else:
        num_query_frames = self.num_frames // stride
        query_endpts = torch.linspace(start, end, num_query_frames + 1)
        query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        print(f"{query_idcs=}")
        for query_idx in tqdm(query_idcs, desc="Loading bkgd points", leave=False):
            img = self.get_image(query_idx)
            feat = self.get_feat(query_idx)
            depth = self.get_depth(query_idx)
            bg_mask = self.get_mask(query_idx) < 0
            bool_mask = (bg_mask * (depth > 0)).to(torch.bool)

            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]

            # get the bounding box of previous points that reproject into frame
            # inefficient but works for now
            bmax_x, bmax_y, bmin_x, bmin_y = 0, 0, W, H
            for p3d, _, _, _ in bg_geometry:
                if len(p3d) < 1:
                    continue
                # reproject into current frame
                p2d = torch.einsum(
                    "ij,jk,pk->pi", K, w2c[:3], F.pad(p3d, (0, 1), value=1.0)
                )
                p2d = p2d[:, :2] / p2d[:, 2:].clamp(min=1e-6)
                xmin, xmax = p2d[:, 0].min().item(), p2d[:, 0].max().item()
                ymin, ymax = p2d[:, 1].min().item(), p2d[:, 1].max().item()

                bmin_x = min(bmin_x, int(xmin))
                bmin_y = min(bmin_y, int(ymin))
                bmax_x = max(bmax_x, int(xmax))
                bmax_y = max(bmax_y, int(ymax))

            # don't include points that are covered by previous points
            bmin_x = max(0, bmin_x)
            bmin_y = max(0, bmin_y)
            bmax_x = min(W, bmax_x)
            bmax_y = min(H, bmax_y)
            overlap_mask = torch.ones_like(bool_mask)
            overlap_mask[bmin_y:bmax_y, bmin_x:bmax_x] = 0

            bool_mask &= overlap_mask
            if bool_mask.sum() < min_per_frame:
                guru.debug(f"skipping {query_idx=}")
                continue

            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]
            point_feats = feat[bool_mask]

            num_sel = max(len(points) // down_rate, min_per_frame)
            sel_idcs = np.random.choice(len(points), num_sel, replace=False)
            points = points[sel_idcs]
            point_normals = point_normals[sel_idcs]
            point_colors = point_colors[sel_idcs]
            point_feats = point_feats[sel_idcs]
            guru.debug(f"{query_idx=} {points.shape=}")
            bg_geometry.append((points, point_normals, point_colors, point_feats))

        bg_points, bg_normals, bg_colors, bg_feats = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )
        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]
            bg_feats = bg_feats[sel_idcs]

        return bg_points, bg_normals, bg_colors, bg_feats

    def load_cam(self, c):
        """Load camera parameters for a given index."""
        md = self.md
        t = self.t
        h, w = md['hw'][c]
        k = md['k'][t][c]
        w2c = np.linalg.inv(md['w2c'][t][c])
        cam = self.setup_camera(w, h, k, w2c, near=self.near, far=self.far)
        return cam

    def load_im(self, c):
        """Load image for a given index."""
        md = self.md
        t = self.t
        fn = md['fn'][t][c]
        try:
          im_path = f"/ssd0/zihanwa3/data_ego/{self.seq}/ims/undist_data/{fn}"
          im = torch.from_numpy(imageio.imread(im_path)).float() / 255.0
        except:
          im_path = f"/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{self.seq}/ims/{fn}"
          im = torch.from_numpy(imageio.imread(im_path)).float() / 255.0
        return im

    def load_im_stat(self, c):
        """Load static image for a given index."""
        md = self.md
        t = self.t
        fn = md['fn'][t][c]
        im_path = f"/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{self.seq}/ims/{fn}"
        im = np.array(Image.open(im_path))
        im = torch.tensor(im).float().permute(2, 0, 1) / 255
        return im

    def load_mask(self, c):
        """Load mask for a given index."""
        md = self.md
        t = self.t
        fn = md['fn'][t][c]

        if self.dino_mask:
            mask_path = f"/ssd0/zihanwa3/data_ego/lalalal_newmask/{fn.split('/')[-1]}"
        else:
            mask_path = f"/ssd0/zihanwa3/data_ego/SR_7_mask/{fn.split('/')[-1].replace('.jpg', '.png')}"


        mask_npz_path = f"/data3/zihanwa3/Capstone-DSR/Appendix/SR_49_clean_mask/{fn.split('/')[-1].replace('.jpg', '.npz')}"
        mask = np.load(mask_npz_path)['mask'][0]

        transform = transforms.ToTensor()
        mask_tensor = transform(mask).squeeze(0)
        return mask_tensor

    def load_depth(self, c):
        """Load depth map for a given index."""
        depth_path = f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
        depth_data = np.load(depth_path)['depth_map']
        depth = torch.tensor(depth_data)
        return depth

    def load_feature(self, c):
        """Load feature map for a given index."""
        md = self.md
        t = self.t
        fn = md['fn'][t][c]
        feature_path = f'/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/bg_feats/{fn.replace(".jpg", ".npy")}'
        dinov2_feature = torch.tensor(np.load(feature_path))
        return dinov2_feature

    def load_depth_stat(self, c):
        """Load static depth map for a given index."""
        depth_path = f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
        depth_data = np.load(depth_path)['depth_map']
        disp_map = depth_data

        nonzero_mask = disp_map != 0
        disp_map[nonzero_mask] = disp_map[nonzero_mask]
        valid_depth_mask = (disp_map > 0) & (disp_map <= self.far)
        disp_map[~valid_depth_mask] = 0
        depth_map = np.full(disp_map.shape, np.inf)
        depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.astype(np.float32)
        depth = torch.tensor(depth_map)
        return depth

    def setup_camera(self, w, h, k, w2c, near, far):
        """Set up camera parameters."""
        # Replace this placeholder with your actual camera setup implementation
        cam = {
            'w': w,
            'h': h,
            'K': k,
            'w2c': w2c,
            'near': near,
            'far': far
        }
        return cam

    # Optional get_ methods to access preloaded data
    def get_cam(self, idx):
        return self.cams[idx]

    def get_Ks(self):
        return torch.cat(self.Ks)

    def get_w2cs(self):
        return torch.cat(self.w2cs)

    def get_im(self, idx):
        return self.images[idx]

    def get_mask(self, idx):
        return self.masks[idx]

    def get_depth(self, idx):
        return self.depths[idx]

    def get_feature(self, idx):
        return self.features[idx]
    



class StatDataset(Dataset):
    def __init__(self, t, md, seq, mode='stat_only', clean_img=True, depth_loss=False, debug_mode='no'):
        self.md = md
        self.seq = seq

        self.near = 1e-7
        self.far = 70.0
        self.indices = []

        self.t = 0
        for lis in [[1400,1401,1402,1403]]:
            for iiiindex, c in sorted(enumerate(lis)):
                self.indices.append(('ego', iiiindex, c))

        # Initialize lists to hold preloaded data
        self.cams = []

        self.Ks=[]
        self.w2cs=[]
        self.images = []
        self.masks = []
        self.depths = []
        self.features = []
        self.ids = []
        self.antimasks = []
        self.visibilities = []
        print("Preloading data...")

        print(len(self.indices), 'leenen')
        for idx, (data_type, iiiindex, c) in enumerate(self.indices):
          cam = self.load_cam(c)
          h, w = 2160, 3840
          H, W = 288, 512
          im = self.load_im(c)
          im = im.permute(2, 0, 1)  # Now im is of shape [3, 2160, 3840]

          # Add batch dimension (needed for interpolation)
          im = im.unsqueeze(0)  # Shape becomes [1, 3, 2160, 3840]

          # Resize the image
          H, W = 288, 512
          resized_im = F.interpolate(im, size=(H, W), mode='bilinear', align_corners=False)

          # Remove the batch dimension
          resized_im = resized_im.squeeze(0)  # Shape is back to [3, 288, 512]

          # Permute back to original shape [H, W, C] if needed
          im = resized_im.permute(1, 2, 0) 



          self.cams.append(cam)

          k = torch.tensor(cam['K']).float()
          sy, sx = H / h, W / w
          fx, fy, cx, cy = k[0][0],  k[1][1], k[0][2], k[1][2], # (4,)

          K = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]]).float() # (3, 3)

          self.Ks.append(K)
          self.w2cs.append(torch.tensor(cam['w2c']).float())

          self.images.append(im)
          self.ids.append(iiiindex)
          self.visibilities.append(True)

          depth = self.load_depth(c)

          dus_depth = self.load_dus_depth(c)
          dus_depth = dus_depth.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 2160, 3840]
          dus_depth = F.interpolate(dus_depth, size=(H, W), mode='bilinear', align_corners=False)
          dus_depth = dus_depth.squeeze(0).squeeze(0)  

          mask = dus_depth > 0

          # Apply the mask to filter out zero values
          filtered_dus_depth = dus_depth[mask]
          filtered_depth = depth[mask]

          # Use nonzero for aligning
          ms_colmap_disp = filtered_dus_depth - torch.median(filtered_dus_depth) + 1e-8
          ms_mono_disp = filtered_depth - torch.median(filtered_depth) + 1e-8


          scale = np.median(ms_colmap_disp / ms_mono_disp)
          depth =  depth *scale
          feature = self.load_feature(c)
          self.depths.append(depth)
          self.features.append(feature)

        print("Data preloading complete.")
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = {}
        cam = self.cams[idx]
        w2cs = self.w2cs[idx].float()
        K = self.Ks[idx].float()

        im = self.images[idx]
        id_ = self.ids[idx]
        data['w2cs'] = w2cs.float()
        data['Ks'] = K.float()
        data['imgs'] = im.float()
        data['id'] = id_
        #data['valid_masks'] = self.antimasks[idx].float()
        data['depths'] = self.depths[idx]
        data['feats'] = self.features[idx].float()

        return data

    def get_bkgd_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = False,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        end = self.num_frames

        print(end, 'frames')
        H, W = self.get_image(0).shape[:2]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )

        ###  start 
        #if use_kf_tstamps:
        #    query_idcs = self.keyframe_idcs.tolist()
        #else:
        num_query_frames = self.num_frames // stride
        query_endpts = torch.linspace(start, end, num_query_frames + 1)
        query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        print(f"{query_idcs=}")
        for query_idx in tqdm(query_idcs, desc="Loading bkgd points", leave=False):
            img = self.get_image(query_idx)
            feat = self.get_feat(query_idx)
            depth = self.get_depth(query_idx)
            bg_mask = self.get_mask(query_idx) < 0
            bool_mask = (bg_mask * (depth > 0)).to(torch.bool)

            w2c = self.w2cs[query_idx]
            K = self.Ks[query_idx]

            # get the bounding box of previous points that reproject into frame
            # inefficient but works for now
            bmax_x, bmax_y, bmin_x, bmin_y = 0, 0, W, H
            for p3d, _, _, _ in bg_geometry:
                if len(p3d) < 1:
                    continue
                # reproject into current frame
                p2d = torch.einsum(
                    "ij,jk,pk->pi", K, w2c[:3], F.pad(p3d, (0, 1), value=1.0)
                )
                p2d = p2d[:, :2] / p2d[:, 2:].clamp(min=1e-6)
                xmin, xmax = p2d[:, 0].min().item(), p2d[:, 0].max().item()
                ymin, ymax = p2d[:, 1].min().item(), p2d[:, 1].max().item()

                bmin_x = min(bmin_x, int(xmin))
                bmin_y = min(bmin_y, int(ymin))
                bmax_x = max(bmax_x, int(xmax))
                bmax_y = max(bmax_y, int(ymax))

            # don't include points that are covered by previous points
            bmin_x = max(0, bmin_x)
            bmin_y = max(0, bmin_y)
            bmax_x = min(W, bmax_x)
            bmax_y = min(H, bmax_y)
            overlap_mask = torch.ones_like(bool_mask)
            overlap_mask[bmin_y:bmax_y, bmin_x:bmax_x] = 0

            bool_mask &= overlap_mask
            if bool_mask.sum() < min_per_frame:
                guru.debug(f"skipping {query_idx=}")
                continue

            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]
            point_feats = feat[bool_mask]

            num_sel = max(len(points) // down_rate, min_per_frame)
            sel_idcs = np.random.choice(len(points), num_sel, replace=False)
            points = points[sel_idcs]
            point_normals = point_normals[sel_idcs]
            point_colors = point_colors[sel_idcs]
            point_feats = point_feats[sel_idcs]
            guru.debug(f"{query_idx=} {points.shape=}")
            bg_geometry.append((points, point_normals, point_colors, point_feats))

        bg_points, bg_normals, bg_colors, bg_feats = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )
        print(bg_points, bg_normals, bg_colors, bg_feats, 'wwwwtf')
        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]
            bg_feats = bg_feats[sel_idcs]

        return bg_points, bg_normals, bg_colors, bg_feats


    def load_cam(self, c):
        """Load camera parameters for a given index."""
        md = self.md
        t = self.t
        h, w = md['hw'][c]
        k = md['k'][t][c]
        w2c = np.linalg.inv(md['w2c'][t][c])
        cam = self.setup_camera(w, h, k, w2c, near=self.near, far=self.far)
        return cam

    def load_im(self, c):
        """Load image for a given index."""
        md = self.md
        t = self.t
        fn = md['fn'][t][c]

        im_path = f"/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/{self.seq}/ims/{fn}"
        im = torch.from_numpy(imageio.imread(im_path)).float() / 255.0
        return im

    def load_dus_depth(self, c):
        """Load depth map for a given index."""
        depth_path = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean/0/conf_depth_{c-1400}.npy'
        depth_data = np.load(depth_path)# ['depth_map']
        depth = torch.tensor(depth_data)
        return depth


    def load_depth(self, c):
        """Load depth map for a given index."""
        depth_path = f'/data3/zihanwa3/Capstone-DSR/any_scripts/stat_depths_512/{c-1399}.npy'
        depth_data = np.load(depth_path)# ['depth_map']
        depth = torch.tensor(depth_data)
        return depth

    def load_feature(self, c):
        """Load feature map for a given index."""
        md = self.md
        t = self.t
        fn = md['fn'][t][c]
        feature_path = f'/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/bg_feats/{fn.replace(".jpg", ".npy")}'
        dinov2_feature = torch.tensor(np.load(feature_path))
        return dinov2_feature

    def load_depth_stat(self, c):
        """Load static depth map for a given index."""
        depth_path = f'/data3/zihanwa3/Capstone-DSR/Processing/da_v2_disp/0/disp_{c}.npz'
        depth_data = np.load(depth_path)['depth_map']
        disp_map = depth_data

        nonzero_mask = disp_map != 0
        disp_map[nonzero_mask] = disp_map[nonzero_mask]
        valid_depth_mask = (disp_map > 0) & (disp_map <= self.far)
        disp_map[~valid_depth_mask] = 0
        depth_map = np.full(disp_map.shape, np.inf)
        depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.astype(np.float32)
        depth = torch.tensor(depth_map)
        return depth

    def setup_camera(self, w, h, k, w2c, near, far):
        """Set up camera parameters."""
        # Replace this placeholder with your actual camera setup implementation
        cam = {
            'w': w,
            'h': h,
            'K': k,
            'w2c': w2c,
            'near': near,
            'far': far
        }
        return cam

    # Optional get_ methods to access preloaded data
    def get_cam(self, idx):
        return self.cams[idx]

    def get_Ks(self):
        return torch.cat(self.Ks)

    def get_w2cs(self):
        return torch.cat(self.w2cs)

    def get_image(self, idx):
        return self.images[idx]

    def get_mask(self, idx):
        return self.masks[idx]

    def get_depth(self, idx):
        return self.depths[idx]

    def get_feat(self, idx):
        return self.features[idx]
    
