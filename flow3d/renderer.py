import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import draw_tracks_2d_th, get_server
from flow3d.vis.viewer import DynamicViewer
from flow3d.params import GaussianParams
import pickle

class Renderer:
    def __init__(
        self,
        model: SceneModel | None = None,
        device: torch.device | None = None,
        # Logging.
        work_dir: str | None = None,
        pc_dir: str | None = None,
        port: int | None = None,
        fg_only: int | None = None,
        seq_name: str | None = 'dance'
    ):
        self.device = device

        self.model = model


        self.pc_dict = {
          'bike': ['/data3/zihanwa3/Capstone-DSR/Processing/dinov2features/', 111],
          'dance': ['/data3/zihanwa3/Capstone-DSR/Processing_dance/dinov2features/', 222],
        }

        self.data_path = self.pc_dict[seq_name][0]
        self.seq_name = seq_name


        self.feat_base = None
        #with open(self.data_path+'fitted_pca_model.pkl', 'rb') as f:
        #  self.feat_base = pickle.load(f)


        if self.model is None:
          self.pc = init_pt_cld = np.load(pc_dir)#["data"]

          self.num_frames = self.pc_dict[seq_name][1]# 111 ##len(self.pc.keys())

          if self.num_frames == 1:
            self.pc = np.load(pc_dir)["data"]
            #self.num_frames = num_copies = 100

            # Create copies and add noise to each slice
            # Assuming the noise is Gaussian with mean 0 and standard deviation 0.01
            #noisy_copies = [self.pc + np.random.normal(0, 0.1, self.pc.shape) for _ in range(num_copies)]

            # Concatenate the noisy copies into a single array of shape [100, ...]
            #self.pc = noisy_pc = np.stack(noisy_copies, axis=0)


        else:
          self.num_frames = model.num_frames

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0



        self.viewer = None
        if self.model is None:
          self.tracks_3d = None
          if port is not None:
              server = get_server(port=port)
              self.viewer = DynamicViewer(
                  server, self.render_fn_no_model, self.num_frames, work_dir, mode="rendering"
              )

        else:
          if port is not None:
              server = get_server(port=port)
              if fg_only:
                self.viewer = DynamicViewer(
                    server, self.render_fn_fg, model.num_frames, work_dir, mode="rendering"
                )
              else:
                self.viewer = DynamicViewer(
                    server, self.render_fn, model.num_frames, work_dir, mode="rendering"
                )
          self.tracks_3d = self.model.compute_poses_fg(
              #  torch.arange(max(0, t - 20), max(1, t), device=self.device),
              torch.arange(self.num_frames, device=self.device),
              inds=torch.arange(49, device=self.device),
          )[0]

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)


        do_my_trick=True
        print(path)
        model_fg=None
        if do_my_trick:
           print('did the')
           if 'piano' in path:
            fg_path = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/results_indiana_piano_14_4/_init_opt_63'
            fg_path = f"{fg_path}/checkpoints/last.ckpt"
            ckpt_fg = torch.load(fg_path)["model"]
            model_fg = SceneModel.init_from_state_dict(ckpt_fg)
            model_fg = model_fg.to(device) 
           elif 'cprrrr' in path:   
            fg_path = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/results_nus_cpr_08_1/_init2'
            fg_path = f"{fg_path}/checkpoints/last.ckpt"
            ckpt_fg = torch.load(fg_path)["model"]
            model_fg = SceneModel.init_from_state_dict(ckpt_fg)
            model_fg = model_fg.to(device) 
           if model_fg is None:
            model.bg = model.bg 
           else:
            model.bg = model_fg.bg#.params
           model.fg.params['opacities'] =  torch.logit(model.fg.params['opacities'] -  model.fg.params['opacities'])
           model.bg.params['scales'] =  1.07 * model.bg.params['scales']


        '''
        def initialize_new_params(new_pt_cld):
            num_pts = new_pt_cld.shape[0]
            means3D = torch.tensor(new_pt_cld[:, :3], dtype=torch.float, device="cuda")  # Convert to torch tensor, shape [num_gaussians, 3]
            unnorm_rots = torch.tensor(np.tile([1, 0, 0, 0], (num_pts, 1)), dtype=torch.float, device="cuda")  # [num_gaussians, 4]
            logit_opacities = torch.zeros((num_pts), dtype=torch.float, device="cuda")  # [num_gaussians, 1]
            
            sq_dist, _ = o3d_knn(new_pt_cld[:, :3], 3)  # Assuming o3d_knn returns a numpy array
            mean3_sq_dist = np.clip(sq_dist.mean(-1), a_min=0.0000001, a_max=None)
            
            seg = np.ones((num_pts))
            seg_colors_np = np.stack((seg, np.zeros_like(seg), 1 - seg), -1)  # [num_pts, 3]

            # Convert everything to torch tensors
            params = {
                'means3D': means3D,  # Already converted
                'rgb_colors': torch.tensor(new_pt_cld[:, 3:6], dtype=torch.float, device="cuda"),  # [num_gaussians, 3]
                'unnorm_rotations': unnorm_rots,  # Already converted
                'seg_colors': torch.tensor(seg_colors_np, dtype=torch.float, device="cuda"),  # [num_pts, 3]
                'logit_opacities': logit_opacities,  # Already converted
                'log_scales': torch.tensor(np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)), dtype=torch.float, device="cuda")  # [num_gaussians, 3]
            }

            return params
        params =
      
        bg_means = params['means3D']#[0]
        bg_quats = params['unnorm_rotations']#[0]
        bkdg_scales = params['log_scales']
        bkdg_colors = params['rgb_colors'] #* 255#[0]
        bg_opacities = params['logit_opacities'][:, 0]
        bg_scene_center = bg_means.mean(0)
        bkdg_feats=torch.ones(bg_means.shape[0], 32)

        gaussians = GaussianParams(
            bg_means,
            bg_quats,
            bkdg_scales,
            bkdg_colors,
            bg_opacities,
        )

        model.bg = Ggaussians
        '''
        renderer = Renderer(model, device, *args, **kwargs)

           
        renderer.global_step = 7000
        renderer.epoch = 499
        print(renderer.global_step, renderer.epoch, 'renderfig')
        return renderer

    @staticmethod
    def init_from_pc_checkpoint(
        device: torch.device, *args, **kwargs
    ) -> "Renderer":
        renderer = Renderer(device=device, *args, **kwargs)
        renderer.global_step = 7000
        renderer.epoch = 499
        return renderer

    @torch.inference_mode()
    def render_fn_no_model(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not self.viewer._canonical_checkbox.value
            else None
        )

        self.seq_name='av'
        if self.seq_name == 'bike':
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing/duster_depth_new_2.7/{t+183}/pc.npz'
          pc = np.load(pc_dir)["data"]
        elif self.seq_name == 'av':
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/AV-proj/street-gaussians-ns/results/baseline_pc/point_cloud_{t}.npy'
          
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/AV-proj/street-gaussians-ns/results/filtered_pc/point_cloud_{t}.npy'
          pc = np.load(pc_dir)
        
        elif self.seq_name == 'see_moge':
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_unc_basketball_03-16-23_01_18/dust_depth/{3*t+210}/pc.npz'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_new_basketball/combined_pointcloud_{3*t+210}.npy'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_new_cooking/combined_pointcloud_{3*t+9000}.npy'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_new_music/combined_pointcloud_{3*t+2820}.npy'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_algo_cpr/combined_pointcloud_{3*t+720}.npy'

          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_cmu_soccer_07_3/dust_depth/{3*t+10050}/pc.npz'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_uniandes_ball_002_17/dust_depth/{3*t+437}/pc.npz'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_indiana_music_14_4/dust_depth/{3*t+1621}/pc.npz'

          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_algo_piano/combined_pointcloud_{3*t+1621}.npy'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_new_softball/combined_pointcloud_{3*t}.npy'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_algo_ball/combined_pointcloud_{3*t+437}.npy'

          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_algo_soccer/combined_pointcloud_{3*t+10070}.npy'

          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/combined_pointclouds_test_new_softball/combined_pointcloud_{3*t}.npy'
          pc = np.load(pc_dir)# ['data']
        elif self.seq_name == 'single_person':
          #t = t * 3 /data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike
          path = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego/cmu_bike/init_pt_cld.npz'
          #path = '/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_300/118/bg_pc.npz'
          new_pt_cld = np.load(path)["data"]
          path = '/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_300/118/bg_pc.npz'
          new_pt_cld_ = np.load(path)["data"]

          print(new_pt_cld.shape, new_pt_cld_.shape)
          pc = np.concatenate((new_pt_cld, new_pt_cld_), axis=0)


        

        elif self.seq_name == 'moving_cam':
          #t = t * 3
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/BIKE_no_preset/{t}/conf_pc.npz'
          pc = np.load(pc_dir)["data"]

          
        elif self.seq_name == 'dance':
          t = t * 3
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_dance/duster_depth_new_2.7/{t+1477}/pc.npz'
          pc = np.load(pc_dir)["data"]
        elif self.seq_name == 'monst3r':
          t = t * 3
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512/{t+1477}/bg_pc.npz'
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/combined_pointclouds/combined_pointcloud_{t+1477}.npy'
          pc = np.load(pc_dir)
        elif self.seq_name == 'dust3r':
          t = t * 3
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_new_pc/{t+1477}/pc.npz'
        elif self.seq_name == 'woc':
          t = t * 3
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_cp/{t+1477}/pc.npz'
          pc = np.load(pc_dir)["data"]

        elif self.seq_name == 'bike_dusmon':
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_bike_100_bs1/{t+183}/conf_pc.npz'
          pc = np.load(pc_dir)['data']
        elif self.seq_name == 'bike_woc':
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/combined_pointclouds_test/combined_pointcloud_{t+183}.npy'
          pc = np.load(pc_dir)

        elif self.seq_name == 'bike_woc_300':
          
          # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/BIKE_duster_depth_clean_300_testonly
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/combined_pointclouds_test_bike/combined_pointcloud_{t+183}.npy'
          pc = np.load(pc_dir)

        elif self.seq_name == 'bike_woc_300_wtf':
          t = t 
          # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/BIKE_duster_depth_clean_300_testonly
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/BIKE_no_preset/{t}/pc.npz'
          pc = np.load(pc_dir)['data']


        elif self.seq_name == 'wtf':
          #t = t * 3
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_monst3r_adj_cao/{t}/pc.npz'
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_clean_dance_512_4_mons_dec/{t+1477}/pc.npz'
          pc = np.load(pc_dir)["data"]
        elif self.seq_name == 'plz':
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/tmp/undist_cam01/{t}/pc.npz'
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_dance/duster_depth_test/{t+1477}/pc.npz'
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/combined_pointclouds/combined_pointcloud_{t+1477}.npy'
          pc = np.load(pc_dir)["data"] 
        elif self.seq_name == 'lllong':
          #/data3/zihanwa3/Capstone-DSR/monst3r/allcat/10/pc.npz
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/allcat/{t}/pc.npz'
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_dance/duster_depth_test/{t+1477}/pc.npz'
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/combined_pointclouds/combined_pointcloud_{t+1477}.npy'
          pc = np.load(pc_dir)["data"] 

        elif self.seq_name == 'last2':
          #/data3/zihanwa3/Capstone-DSR/monst3r/allcat/10/pc.npz
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/duster_depth_monst3r/{t}/pc.npz'
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_dance/duster_depth_test/{t+1477}/pc.npz'
          #pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/combined_pointclouds/combined_pointcloud_{t+1477}.npy'
          pc = np.load(pc_dir)["data"] 


        pc = torch.tensor(pc).cuda()[:, :6].float()
        #pc[:, 3:] = pc[:, 3:] / 255
        #except:
        #  print(self.pc.shape)
        #  pc = torch.tensor(self.pc).cuda()[:, :6].float()

        pc_xyz = pc[:, :3]   # Shape: (N, 3)
        colors = pc[:, 3:6]  # Shape: (N, 3)

        # Create homogeneous coordinates for XYZ
        ones = torch.ones((pc_xyz.shape[0], 1), device=pc.device)
        pc_homogeneous = torch.cat([pc_xyz, ones], dim=1)  # Shape: (N, 4)

        # Transform points to camera coordinates
        pc_camera = (w2c @ pc_homogeneous.T).T  # Shape: (N, 4)

        # Discard the homogeneous coordinate
        pc_camera = pc_camera[:, :3]  # Shape: (N, 3)

        X_c = pc_camera[:, 0]
        Y_c = pc_camera[:, 1]
        Z_c = pc_camera[:, 2]

        # Avoid division by zero
        Z_c[Z_c == 0] = 1e-6

        # Project onto image plane
        x = X_c / Z_c
        y = Y_c / Z_c

        u = K[0, 0] * x + K[0, 2]
        v = K[1, 1] * y + K[1, 2]

        # Image dimensions
        width, height = W, H

        # Validity mask
        valid_mask = (Z_c > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        u = u[valid_mask]
        v = v[valid_mask]
        colors = colors[valid_mask]  # Apply mask to colors

        # Initialize an empty image
        img = torch.zeros((height, width, 3), device=pc.device) 
        u_int = u.long()
        v_int = v.long()
        img[v_int, u_int] = colors

        if not self.viewer._render_track_checkbox.value:
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)

        else:
            assert t is not None
            tracks_3d = self.tracks_3d[:, max(0, t - 20) : max(1, t)]
            tracks_2d = torch.einsum(
                "ij,jk,nbk->nbi", K, w2c[:3], F.pad(tracks_3d, (0, 1), value=1.0)
            )
            tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
            img = draw_tracks_2d_th(img, tracks_2d)
        return img

    @torch.inference_mode()
    def render_fn_fg(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        cx, cy =  W / 2.0, H / 2.0
        cx -= 0.5
        cy -= 0.5
        K = torch.tensor(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not self.viewer._canonical_checkbox.value
            else None
        )
        self.model.training = False
        #fg_only=True
        img = self.model.render(t, w2c[None], K[None], img_wh, fg_only=False)["img"][0]
        feat = self.model.render(t, w2c[None], K[None], img_wh, fg_only=False)["feat"][0]



        if not self.viewer._render_track_checkbox.value:
            if self.feat_base:
              # feat: torch.Size([1029, 2048, 32])
              pca_features = self.feat_base.transform(feat.cpu().numpy().reshape(-1, 32))
              #print(pca_feat.shape, img.cpu().numpy().min(), img.cpu().numpy().max(),)
              #pca_feat = pca_feat.reshape(feat.shape[0], feat.shape[1], -1)
              #print(pca_feat.shape, pca_feat.min(), pca_feat.max())
              pca_features_norm = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
              pca_features_norm = (pca_features_norm * 255).astype(np.uint8)
              
              # Reconstruct full image
              full_pca_features = np.zeros((pca_features.shape[0], 3), dtype=np.uint8)
              #if mask is not None:
              #    full_pca_features[mask_flat] = pca_features_norm
              #else:
              full_pca_features = pca_features_norm
              pca_features_image = full_pca_features.reshape(feat.shape[0], feat.shape[1], 3)
              
              return pca_features_image

            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        else:
            assert t is not None
            tracks_3d = self.tracks_3d[:, max(0, t - 20) : max(1, t)]
            tracks_2d = torch.einsum(
                "ij,jk,nbk->nbi", K, w2c[:3], F.pad(tracks_3d, (0, 1), value=1.0)
            )
            tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
            img = draw_tracks_2d_th(img, tracks_2d)
        return img


    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        cx, cy =  W / 2.0, H / 2.0
        cx -= 0.5
        cy -= 0.5
        K = torch.tensor(
            [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not self.viewer._canonical_checkbox.value
            else None
        )
        self.model.training = False
        #fg_only=True
        ## torch.Size([1, 1029, 2048, 32]) -> torch.Size([1029, 2048, 3])
        img = self.model.render(t, w2c[None], K[None], img_wh, )["img"][0]
        feat = self.model.render(t, w2c[None], K[None], img_wh, )["feat"][0]

        
        #[0]

        if not self.viewer._render_track_checkbox.value:
            if self.feat_base:
              # feat: torch.Size([1029, 2048, 32])
              pca_features = self.feat_base.transform(feat.cpu().numpy().reshape(-1, 32))
              #print(pca_feat.shape, img.cpu().numpy().min(), img.cpu().numpy().max(),)
              #pca_feat = pca_feat.reshape(feat.shape[0], feat.shape[1], -1)
              #print(pca_feat.shape, pca_feat.min(), pca_feat.max())
              pca_features_norm = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
              pca_features_norm = (pca_features_norm * 255).astype(np.uint8)
              
              # Reconstruct full image
              full_pca_features = np.zeros((pca_features.shape[0], 3), dtype=np.uint8)
              #if mask is not None:
              #    full_pca_features[mask_flat] = pca_features_norm
              #else:
              full_pca_features = pca_features_norm
              pca_features_image = full_pca_features.reshape(feat.shape[0], feat.shape[1], 3)
              
              return pca_features_image
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
            
        else:
            assert t is not None
            tracks_3d = self.tracks_3d[:, max(0, t - 20) : max(1, t)]
            tracks_2d = torch.einsum(
                "ij,jk,nbk->nbi", K, w2c[:3], F.pad(tracks_3d, (0, 1), value=1.0)
            )
            tracks_2d = tracks_2d[..., :2] / tracks_2d[..., 2:]
            img = draw_tracks_2d_th(img, tracks_2d)
        return img