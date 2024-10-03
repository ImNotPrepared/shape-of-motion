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
          print(pc_dir, self.pc.keys(), self.num_frames)

          if self.num_frames == 1:
            self.pc = np.load(pc_dir)["data"]

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
    
    @staticmethod
    def project_pointcloud_to_image(pc_xyz, colors, K, w2c, H, W, device):
        # pc_xyz: (N, 3)
        # colors: (N, 3)
        # K: (3, 3)
        # w2c: (4, 4)
        # H, W: image dimensions
        # device: torch device

        N = pc_xyz.shape[0]

        # Step 1: Transform to homogeneous coordinates
        ones = torch.ones((N, 1), device=device)
        pc_xyz_h = torch.cat([pc_xyz, ones], dim=1)  # (N, 4)

        # Step 2: Apply world-to-camera transformation
        pc_xyz_cam = (w2c @ pc_xyz_h.T).T[:, :3]  # (N, 3)

        x_cam = pc_xyz_cam[:, 0]
        y_cam = pc_xyz_cam[:, 1]
        z_cam = pc_xyz_cam[:, 2]

        # Only keep points in front of the camera
        valid_mask = z_cam > 0
        x_cam = x_cam[valid_mask]
        y_cam = y_cam[valid_mask]
        z_cam = z_cam[valid_mask]
        colors = colors[valid_mask]

        # Step 3: Project onto image plane using intrinsic parameters
        focal = K[0, 0]
        u = (focal * x_cam / z_cam) + K[0, 2]
        v = (focal * y_cam / z_cam) + K[1, 2]

        # Convert to integer pixel indices
        u_int = u.long()
        v_int = v.long()

        # Keep points within image bounds
        in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        u_int = u_int[in_bounds]
        v_int = v_int[in_bounds]
        z_cam = z_cam[in_bounds]
        colors = colors[in_bounds]

        # Compute linear pixel indices
        pixel_indices = v_int * W + u_int  # Shape: (M,)

        # Step 4: Handle depth ordering using sorting
        sort_indices = torch.argsort(pixel_indices)
        pixel_indices_sorted = pixel_indices[sort_indices]
        z_cam_sorted = z_cam[sort_indices]
        colors_sorted = colors[sort_indices]

        # Find unique pixels and their first occurrence (closest point)
        unique_pixels, first_occurrence_indices = torch.unique_consecutive(
            pixel_indices_sorted, return_indices=True
        )

        # Step 5: Initialize depth buffer and image buffer
        depth_buffer = torch.full((H * W,), float('inf'), device=device)
        image_buffer = torch.zeros((H * W, 3), device=device)

        # Update buffers with closest point data
        depth_buffer[unique_pixels] = z_cam_sorted[first_occurrence_indices]
        image_buffer[unique_pixels] = colors_sorted[first_occurrence_indices]
        image = image_buffer.view(H, W, 3)

        return image

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
        pc = torch.tensor(self.pc[str(t)]).cuda()[:, :6].float()
        #except:
        # pc = torch.tensor(self.pc).cuda()[:, :6].float()

        pc_xyz = pc[:, :3]   # Shape: (N, 3)
        colors = pc[:, 3:6]  # Shape: (N, 3)

        img = self.project_pointcloud_to_image(pc_xyz, colors, K, w2c, H, W, 'cuda')

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
        print(t)
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