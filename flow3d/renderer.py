import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import draw_tracks_2d_th, get_server
from flow3d.vis.viewer import DynamicViewer
from flow3d.params import GaussianParams, GaussianParamsOthers
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
          'dance': ['/data3/zihanwa3/Capstone-DSR/Processing_dance/dinov2features/', 100],
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
          if self.model.fg is not None:
            self.num_frames = model.num_frames
          else:
            self.num_frames = 1


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
              if self.model.fg is not None:
                if fg_only:
                  self.viewer = DynamicViewer(
                      server, self.render_fn_fg, self.num_frames, work_dir, mode="rendering"
                  )
                else:
                  self.viewer = DynamicViewer(
                      server, self.render_fn, self.num_frames, work_dir, mode="rendering"
                  )
              else: 
                  self.viewer = DynamicViewer(
                      server, self.render_fn_bg, self.num_frames, work_dir, mode="rendering"
                  )

          if self.model.fg is not None:
            self.tracks_3d = self.model.compute_poses_fg(
                #  torch.arange(max(0, t - 20), max(1, t), device=self.device),
                torch.arange(self.num_frames, device=self.device),
                inds=torch.arange(49, device=self.device),
            )[0]
          else:
            self.tracks_3d = None 

    @staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> "Renderer":
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)

        # print('CAUSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS_DOTHETRICKS')

        '''
        def get_scales(self) -> torch.Tensor:
          return self.scale_activation(self.params["scales"])

        rendervar = {
            'means3D': params['means3D'],
            'colors_precomp': params['rgb_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
            'semantic_feature': params['semantic_feature'],
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0,
            'label': params['label']

        }

        
        '''
        ckpt_path='/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/output/1_conf_bg_new_clip/cmu_bike/params_iter_10000.npz'

        params = dict(np.load(ckpt_path, allow_pickle=True))

        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        means = params['means3D']#[0]
        quats = params['unnorm_rotations']#[0]
        scales = params['log_scales']
        colors = params['rgb_colors'] #* 10#* 255#[0]
        opacities = (params['logit_opacities'][:, 0])#.clip(0, 1)


        if model.fg is not None:
          path = 'resultsbike_bg_only/clean_1_no_feat_nodensify/checkpoints/last.ckpt'
          #path =         ckpt_path='/data3/zihanwa3/Capstone-DSR/monst3r_train/my_data_2/Dynamic3DGaussians/output/official/basketball/params.npz'
          state_dict =  torch.load(path)["model"]
          model_bg = SceneModel.init_from_state_dict(state_dict)
          model_bg = model_bg.to(device)
          model.bg = model_bg.bg
        model.bg.feats = None 
        

          #model.bg = GaussianParamsOthers(means, quats, scales, colors, opacities)
        do_my_trick=True
        if do_my_trick:
           model.bg.params['scales'] = 0.93 * model.bg.params['scales']

        for k, v in model.bg.params.items():
           print(k, v.mean())
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
        # [t]
        # Assuming self.pc is of shape [N, 6] with XYZRGB
        #try:  
          # pc = torch.tensor(self.pc[str(t)]).cuda()[:, :6].float()
        # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing/duster_depth_new/{t+183}/fg_pc.npz'

        self.seq_name='panoptic'
        if self.seq_name == 'bike':
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing/duster_depth_new_2.7/{t+183}/pc.npz'
          pc = np.load(pc_dir)["data"]

        elif self.seq_name == 'panoptic':
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/monst3r/combined_pointclouds_test_new_softball/combined_pointcloud_{3*t}.npy'
          # pc_dir = f'/data3/zihanwa3/Capstone-DSR/Processing_panoptic_dense_check_tennis/dumon_depth/{t}/pc.npz'
          pc = np.load(pc_dir)#["data"]


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

        elif self.seq_name == 'moge':
          #/data3/zihanwa3/Capstone-DSR/monst3r/allcat/10/pc.npz
          if (3*t+183)>296:
            t=t-1
          pc_dir = f'/data3/zihanwa3/Capstone-DSR/Appendix/MoGe/moge_pc/moge_combined_pointcloud_{3*t+183}.npy'
          pc = np.load(pc_dir)



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
    def render_fn_bg(self, camera_state: CameraState, img_wh: tuple[int, int]):
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
        t = None
        self.model.training = False
        img = self.model.render_stat_bg(t, w2c[None], K[None], img_wh, )["img"][0]
        # feat = self.model.render(t, w2c[None], K[None], img_wh, )["feat"][0]

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
        # feat = self.model.render(t, w2c[None], K[None], img_wh, )["feat"][0]

        
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