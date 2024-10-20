import os
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from roma import roma
from tqdm import tqdm

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
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 7
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False
    video_name: str = ''



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
        depth_type: Literal[
            "aligned_depth_anything",
            "aligned_depth_anything_v2",
            "depth_anything",
            "depth_anything_v2",
            "unidepth_disp",
        ] = "aligned_depth_anything",
        camera_type: Literal["droid_recon"] = "droid_recon",
        track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir",
        mask_erosion_radius: int = 3,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        video_name: str = '',
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

        # self.camera_path
        self.video_name = video_name# '_dance'
        self.hard_indx_dict = {
          '_dance': [1477, 1778],
          '': [183, 295],
        }
        self.glb_first_indx =  self.hard_indx_dict[self.video_name][0]
        self.glb_last_indx = self.hard_indx_dict[self.video_name][1]

        self.depth_dir = f"{root_dir}/{depth_type}/{res}/{seq_name}"
        self.mask_dir = f"{root_dir}/{mask_type}/{res}/{seq_name}"
        self.tracks_dir = f"{root_dir}/{track_2d_type}/{res}/{seq_name}"
        self.cache_dir = f"{root_dir}/flow3d_preprocessed/{res}/{seq_name}"


        
        #  self.cache_dir = f"datasets/davis/flow3d_preprocessed/{res}/{seq_name}"
        frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir))]

        if end == -1:
            end = len(frame_names)
        self.start = start
        self.end = end
        self.frame_names = frame_names[start:end]
        print(self.start, self.end)

        self.imgs: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.feats: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.depths: list[torch.Tensor | None] = [None for _ in self.frame_names]
        self.masks: list[torch.Tensor | None] = [None for _ in self.frame_names]



        self.debug=False
        def load_known_cameras(
            path: str, H: int, W: int, noise: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert os.path.exists(path), f"Camera file {path} does not exist."
            md = json.load(open(path, 'r'))
            c2ws = []
            #for c in range(4, 5):
            c = int(self.seq_name[-1])
            for t in range(self.glb_first_indx, self.glb_last_indx):
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
            path = f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/scripts/Dy_train_meta.json'
            if self.debug:
              w2cs, Ks, tstamps = load_cameras(
                  f"{root_dir}/{camera_type}/{seq_name}.npy", H, W
              )
            else:
              w2cs, Ks, tstamps = load_known_cameras(
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

    def get_depth(self, index) -> torch.Tensor:
        if self.depths[index] is None:
            self.depths[index] = self.load_depth(index)
        return self.depths[index] #/ self.scale

    def load_image(self, index) -> torch.Tensor:
        path = f"{self.img_dir}/{self.frame_names[index]}{self.img_ext}"
        return torch.from_numpy(imageio.imread(path)).float() / 255.0

    def load_feat(self, index) -> torch.Tensor:
        # path = f"{self.feat_dir}/{self.frame_names[index]}{self.feat_ext}"¸¸
        path = f"{self.feat_dir}/{self.frame_names[index]}{self.feat_ext}"
        ### examples:
        # /data3/zihanwa3/Capstone-DSR/Processing_dance/dinov2features/resized_512_Aligned_fg_only/undist_cam01 
        #
        # 
        try:
          path = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/dinov2features/resized_512_Aligned/')
          path = path.replace('toy_512_', 'undist_cam0') # cam0x              ############# _fg_only                                                              _fg_only/
          path = path.replace('jpg', 'npy')
          dinov2_feature = torch.tensor(np.load(path)).to(torch.float32)
        except:
          path = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/dinov2features/resized_512_Aligned_fg_only/')
          path = path.replace('toy_512_', 'undist_cam0') # cam0x              ############# _fg_only                                                              _fg_only/
          path = path.replace('jpg', 'npy')
          dinov2_feature = torch.tensor(np.load(path)).to(torch.float32)          
        # /data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images//toy_512_1/00183.jpg
        # 
        # /data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512_registered/undist_cam02/00000.npy
        #feat
        #feature_root_path='/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/resized_512/' #undist_cam00_670/000000.npy'
        #feature_path = feature_root_path+fn 
        #print(path)
        dinov2_feature = torch.tensor(np.load(path)).to(torch.float32)#.permute(2, 0, 1)
        #print(dinov2_feature.dtype, 'DDDDDtpye')
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
            path = f"{self.mask_dir}/dyn_mask_{int(self.frame_names[index])}.npz"
            r = self.mask_erosion_radius
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

    def load_depth(self, index) -> torch.Tensor:
        #  load_da2_depth load_duster_depth load_org_depth
        return self.load_duster_depth(index)

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
        except:
          #/data3/zihanwa3/Capstone-DSR/Processing_dance/duster_depth_new_2.7/undist_cam01/1477.npz

          # /data3/zihanwa3/Capstone-DSR/Processing_dance/duster_depth_new_2.7/1478/1.npz
          final_path = f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/duster_depth_new_2.7/' +  path.split('/')[-1][:-4] + '/' + path.split('/')[-2][-1] + '.npz'
          disp_map =  np.load(final_path)['depth']
        depth_map = np.clip(disp_map, a_min=1e-8, a_max=1e6)
        depth = torch.from_numpy(depth_map).float()
        input_tensor = depth.unsqueeze(0).unsqueeze(0) 

        output_size = (288, 512)  # (height, width)
        resized_tensor = F.interpolate(input_tensor, size=output_size, mode='bilinear', align_corners=False)

        # If you want to remove the added dimensions
        depth = resized_tensor.squeeze(0).squeeze(0) 
        return depth



    def load_da2_depth(self, index) -> torch.Tensor:
        path = f"{self.depth_dir}/{self.frame_names[index]}.npy"
        near, far = 1e-7, 7e1
        path = f"{self.depth_dir}/disp_{int(self.frame_names[index])}.npz"
        path = path.replace('/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/aligned_depth_anything//', f'/data3/zihanwa3/Capstone-DSR/Processing{self.video_name}/da_v2_disp/')
        path = path.replace('toy_512_', '')
        camera_index = int(self.depth_dir[-1]) -1 
        absolute_index=index+60 ### now it is 123, 423
        scale, shift = self.scales_shifts[absolute_index][camera_index]
        disp_map =  np.load(path)['depth_map']
        nonzero_mask = disp_map != 0
        
        disp_map[nonzero_mask] = disp_map[nonzero_mask]* scale + shift
        valid_depth_mask = (disp_map > 0) & (disp_map <= far)
        disp_map[~valid_depth_mask] = 0
        depth_map = np.full(disp_map.shape, np.inf)
        depth_map[disp_map != 0] = 1 / disp_map[disp_map != 0]
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.astype(np.float32)
        depth = torch.from_numpy(depth_map).float()
        input_tensor = depth.unsqueeze(0).unsqueeze(0)  # Now size [1, 1, 288, 512]

        # Define the target size
        output_size = (288, 512)  # (height, width)

        # Resize the tensor
        resized_tensor = F.interpolate(input_tensor, size=output_size, mode='bilinear', align_corners=False)
        depth = resized_tensor.squeeze(0).squeeze(0) 
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
        #print('wtttf', colors.shape)
        #print('wtttf', feats.shape)

        return tracks_3d, visibles, invisibles, confidences, colors, feats



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

    def __getitem__(self, index: int):
        index = np.random.randint(0, self.num_frames)
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
        target_inds = torch.from_numpy(
            np.random.choice(
                self.num_frames, (self.num_targets_per_frame,), replace=False
            )
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

