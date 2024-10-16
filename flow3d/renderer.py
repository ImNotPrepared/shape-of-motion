import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState

from flow3d.scene_model import SceneModel
from flow3d.vis.utils import draw_tracks_2d_th, get_server
from flow3d.vis.viewer import DynamicViewer


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
    ):
        self.device = device

        self.model = model
        if self.model is None:
          self.pc = init_pt_cld = np.load(pc_dir)#["data"]
          self.num_frames = len(self.pc.keys())

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
              inds=torch.arange(10, device=self.device),
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
        try:  
          pc = torch.tensor(self.pc[str(t)]).cuda()[:, :6].float()
        except:
          print(self.pc.shape)
          pc = torch.tensor(self.pc).cuda()[:, :6].float()

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
        self.model.training = False
        #fg_only=True
        img = self.model.render(t, w2c[None], K[None], img_wh, fg_only=True)["img"][0]
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
        img = self.model.render(t, w2c[None], K[None], img_wh, )["img"][0]
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