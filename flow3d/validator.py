import functools
import os
import os.path as osp
import time
from dataclasses import asdict
from typing import cast

import imageio as iio
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState, Viewer
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data.utils import normalize_coords, to_device
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import (
    apply_depth_colormap,
    make_video_divisble,
    plot_correspondences,
)


class Validator:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        train_loader: DataLoader | None = None,
        val_img_loader: DataLoader | None = None,
        val_kpt_loader: DataLoader | None = None,
        save_dir: str | None = None,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_img_loader = val_img_loader
        self.val_kpt_loader = val_kpt_loader
        self.save_dir = save_dir
        self.has_bg = self.model.has_bg
        

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS().to(device)
        self.fg_psnr_metric = mPSNR()
        self.fg_ssim_metric = mSSIM()
        self.fg_lpips_metric = mLPIPS().to(device)
        self.bg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS().to(device)
        self.pck_metric = PCK()

    def reset_metrics(self):
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.lpips_metric.reset()
        self.fg_psnr_metric.reset()
        self.fg_ssim_metric.reset()
        self.fg_lpips_metric.reset()
        self.bg_psnr_metric.reset()
        self.bg_ssim_metric.reset()
        self.bg_lpips_metric.reset()
        self.pck_metric.reset()

    @torch.no_grad()
    def validate(self):
        self.reset_metrics()
        metric_imgs = self.validate_imgs() #or {}
        return {**metric_imgs}

    @torch.no_grad()
    def validate_end(self):
        self.reset_metrics()
        metric_imgs = self.validate_trainview() or {}
        return {**metric_imgs}
    

    @torch.no_grad()
    def validate_trainview(self):
        guru.info("rendering validation images...")
        if self.val_img_loader is None:
            return
        for batch in tqdm(self.val_img_loader, desc="render val images"):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0]
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]
            # (1, H, W).
            valid_mask = batch.get(
                "valid_masks", torch.ones_like(batch["imgs"][..., 0])
            )
            # (1, H, W).
            fg_mask = batch["masks"]

            # (H, W).
            covisible_mask = batch.get(
                "covisible_masks",
                torch.ones_like(fg_mask)[None],
            )
            W, H = img_wh = img[0].shape[-2::-1]


            rendered = self.model.render(t, w2c, K, img_wh, return_depth=True)
            valid_mask *= covisible_mask
            fg_valid_mask = fg_mask * valid_mask
            bg_valid_mask = (1 - fg_mask) * valid_mask
            main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)
                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

            # Dump results.
            results_dir = osp.join(self.save_dir, "results", "rgb")
            os.makedirs(results_dir, exist_ok=True)
            iio.imwrite(
                osp.join(results_dir, f"{frame_name}.png"),
                (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
            )

        return {
            "val/psnr": self.psnr_metric.compute(),
            "val/ssim": self.ssim_metric.compute(),
            "val/lpips": self.lpips_metric.compute(),
            "val/fg_psnr": self.fg_psnr_metric.compute(),
            "val/fg_ssim": self.fg_ssim_metric.compute(),
            "val/fg_lpips": self.fg_lpips_metric.compute(),
            "val/bg_psnr": self.bg_psnr_metric.compute(),
            "val/bg_ssim": self.bg_ssim_metric.compute(),
            "val/bg_lpips": self.bg_lpips_metric.compute(),
        }
    

    @torch.no_grad()
    def validate_NTS(self):
        guru.info("rendering validation images...")
        if self.val_img_loader is None:
            return
        for batch in tqdm(self.val_img_loader, desc="render val images"):
            batch = to_device(batch, self.device)
            frame_name = batch["frame_names"][0]
            t = batch["ts"][0]
            # (1, 4, 4).
            w2c = batch["w2cs"]
            # (1, 3, 3).
            K = batch["Ks"]
            # (1, H, W, 3).
            img = batch["imgs"]
            # (1, H, W).
            valid_mask = batch.get(
                "valid_masks", torch.ones_like(batch["imgs"][..., 0])
            )
            # (1, H, W).
            fg_mask = batch["masks"]

            # (H, W).
            covisible_mask = batch.get(
                "covisible_masks",
                torch.ones_like(fg_mask)[None],
            )
            # W, H = img_wh = img[0].shape[-2::-1]

            
            rendered = self.model.render(t, w2c, K, img_wh, return_depth=True)
            valid_mask *= covisible_mask
            fg_valid_mask = fg_mask * valid_mask
            bg_valid_mask = (1 - fg_mask) * valid_mask
            main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)
                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

            # Dump results.
            results_dir = osp.join(self.save_dir, "results", "rgb")
            os.makedirs(results_dir, exist_ok=True)
            iio.imwrite(
                osp.join(results_dir, f"{frame_name}.png"),
                (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
            )

        return {
            "val/psnr": self.psnr_metric.compute(),
            "val/ssim": self.ssim_metric.compute(),
            "val/lpips": self.lpips_metric.compute(),
            "val/fg_psnr": self.fg_psnr_metric.compute(),
            "val/fg_ssim": self.fg_ssim_metric.compute(),
            "val/fg_lpips": self.fg_lpips_metric.compute(),
            "val/bg_psnr": self.bg_psnr_metric.compute(),
            "val/bg_ssim": self.bg_ssim_metric.compute(),
            "val/bg_lpips": self.bg_lpips_metric.compute(),
        }
    

    @torch.no_grad()
    def validate_imgs(self):
        guru.info("rendering validation images...")
        if self.val_img_loader is None:
            return

        # for batch in tqdm(self.val_img_loader, desc="render val images"):
        for batch_idx, batch in enumerate(
            tqdm(self.val_img_loader, desc="Rendering video", leave=False)
        ):
                        
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"]
            # (H, W).
            depth = batch["depths"]

            mask = batch["masks"]

            img_wh = img[0].shape[-2::-1]


            frame_name = batch["frame_names"][0]

            valid_mask = batch.get(
                "valid_masks", torch.ones_like(batch["imgs"][..., 0])
            )
            # (1, H, W).
            fg_mask = batch["masks"]

            # (H, W).
            covisible_mask = batch.get(
                "covisible_masks",
                torch.ones_like(fg_mask),
            )
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )
            #WITHOUT 0 orch.Size([1, 288, 512, 3]) torch.Size([1, 288, 512]) torch.Size([1, 1, 288, 512]) torch.Size([1, 288, 512])   
            #WITH 0 orch.Size([1, 288, 512, 3]) torch.Size([1, 288, 512]) torch.Size([1, 288, 512]) torch.Size([288, 512]) 
            # print(rendered["img"].shape, valid_mask.shape, covisible_mask.shape, fg_mask.shape)
            valid_mask *= covisible_mask
            fg_valid_mask = fg_mask * valid_mask
            bg_valid_mask = (1 - fg_mask) * valid_mask
            main_valid_mask = valid_mask if self.has_bg else fg_valid_mask

            self.psnr_metric.update(rendered["img"], img, main_valid_mask)
            self.ssim_metric.update(rendered["img"], img, main_valid_mask)
            self.lpips_metric.update(rendered["img"], img, main_valid_mask)

            if self.has_bg:
                self.fg_psnr_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_ssim_metric.update(rendered["img"], img, fg_valid_mask)
                self.fg_lpips_metric.update(rendered["img"], img, fg_valid_mask)
                self.bg_psnr_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_ssim_metric.update(rendered["img"], img, bg_valid_mask)
                self.bg_lpips_metric.update(rendered["img"], img, bg_valid_mask)

            # Dump results.
            results_dir = osp.join(self.save_dir, "results", "rgb")
            os.makedirs(results_dir, exist_ok=True)
            iio.imwrite(
                osp.join(results_dir, f"{frame_name}.png"),
                (rendered["img"][0].cpu().numpy() * 255).astype(np.uint8),
            )

        return {
            "val/psnr": self.psnr_metric.compute(),
            "val/ssim": self.ssim_metric.compute(),
            "val/lpips": self.lpips_metric.compute(),
            "val/fg_psnr": self.fg_psnr_metric.compute(),
            "val/fg_ssim": self.fg_ssim_metric.compute(),
            "val/fg_lpips": self.fg_lpips_metric.compute(),
            "val/bg_psnr": self.bg_psnr_metric.compute(),
            "val/bg_ssim": self.bg_ssim_metric.compute(),
            "val/bg_lpips": self.bg_lpips_metric.compute(),
        }


    @torch.no_grad()
    def save_train_videos(self, epoch: int):
        if self.train_loader is None:
            return
        video_dir = osp.join(self.save_dir, "videos", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)
        fps = 15.0

        # Create directories to save individual frames
        image_dir = osp.join(video_dir, "images")
        depth_dir = osp.join(video_dir, "depths")
        mask_dir = osp.join(video_dir, "masks")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # Render video.
        video = []
        ref_pred_depths = []
        masks = []
        depth_min, depth_max = 1e6, 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]
            # (H, W).
            depth = batch["depths"][0]

            mask = batch["masks"][0]

            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            combined_img = torch.cat([img, rendered["img"][0]], dim=1).cpu()
            video.append(combined_img)

            # Save individual RGB images
            rgb_image = (combined_img.numpy() * 255).astype(np.uint8)
            rgb_image = make_video_divisble(rgb_image)
            rgb_image_path = osp.join(image_dir, f"frame_{batch_idx:04d}.png")
            iio.imwrite(rgb_image_path, rgb_image)

            ref_pred_depth = torch.cat(
                (depth[..., None], rendered["depth"][0]), dim=1
            ).cpu()
            ref_pred_depths.append(ref_pred_depth)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())

            # Save individual depth maps
            depth_colormap = apply_depth_colormap(
                ref_pred_depth,
                near_plane=depth_min,
                far_plane=depth_max
            )
            depth_image = (depth_colormap.numpy() * 255).astype(np.uint8)
            depth_image = make_video_divisble(depth_image)
            depth_image_path = osp.join(depth_dir, f"depth_{batch_idx:04d}.png")
            iio.imwrite(depth_image_path, depth_image)

            if rendered["mask"] is not None:
                combined_mask = torch.cat(
                    [mask, rendered["mask"][0].squeeze(-1)], dim=1
                ).cpu()
                masks.append(combined_mask)

                # Save individual masks
                mask_image = (combined_mask.numpy() * 255).astype(np.uint8)
                mask_image = make_video_divisble(mask_image)
                mask_image_path = osp.join(mask_dir, f"mask_{batch_idx:04d}.png")
                iio.imwrite(mask_image_path, mask_image)

        # rgb video
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "rgbs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        # depth video
        depth_video = torch.stack(
            [
                apply_depth_colormap(
                    ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                )
                for ref_pred_depth in ref_pred_depths
            ],
            dim=0,
        )
        iio.mimwrite(
            osp.join(video_dir, "depths.mp4"),
            make_video_divisble((depth_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        if len(masks) > 0:
            # mask video
            mask_video = torch.stack(masks, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "masks.mp4"),
                make_video_divisble((mask_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )

        # Render 2D track video.
        tracks_2d, target_imgs = [], []
        sample_interval = 10
        batch0 = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self.train_loader.dataset[0].items()
        }
        # ().
        t = batch0["ts"]
        # (4, 4).
        w2c = batch0["w2cs"]
        # (3, 3).
        K = batch0["Ks"]
        # (H, W, 3).
        img = batch0["imgs"]
        # (H, W).
        bool_mask = batch0["masks"] > 0.5
        img_wh = img.shape[-2::-1]
        for batch in tqdm(
            self.train_loader, desc="Rendering 2D track video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            # (1, H, W, 3).
            target_imgs.append(batch["imgs"].cpu())
            # (1,).
            target_ts = batch["ts"]
            # (1, 4, 4).
            target_w2cs = batch["w2cs"]
            # (1, 3, 3).
            target_Ks = batch["Ks"]
            rendered = self.model.render(
                t,
                w2c[None],
                K[None],
                img_wh,
                target_ts=target_ts,
                target_w2cs=target_w2cs,
            )
            pred_tracks_3d = rendered["tracks_3d"][0][
                ::sample_interval, ::sample_interval
            ][bool_mask[::sample_interval, ::sample_interval]].swapaxes(0, 1)
            pred_tracks_2d = torch.einsum("bij,bpj->bpi", target_Ks, pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., 2:], min=1e-6
            )
            tracks_2d.append(pred_tracks_2d.cpu())
        tracks_2d = torch.cat(tracks_2d, dim=0)
        target_imgs = torch.cat(target_imgs, dim=0)
        track_2d_video = plot_correspondences(
            target_imgs.numpy(),
            tracks_2d.numpy(),
            query_id=cast(int, t),
        )
        iio.mimwrite(
            osp.join(video_dir, "tracks_2d.mp4"),
            make_video_divisble(np.stack(track_2d_video, 0)),
            fps=fps,
        )
        # Render motion coefficient video.
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            motion_coef_colors = torch.pca_lowrank(
                self.model.fg.get_coefs()[None],
                q=3,
            )[0][0]
        motion_coef_colors = (motion_coef_colors - motion_coef_colors.min(0)[0]) / (
            motion_coef_colors.max(0)[0] - motion_coef_colors.min(0)[0]
        )

        if self.model.bg is None: 
          motion_coef_colors = motion_coef_colors
        else:
          motion_coef_colors = F.pad(
              motion_coef_colors, (0, 0, 0, self.model.bg.num_gaussians), value=0.5
          )
        video = []
        for batch in tqdm(
            self.train_loader, desc="Rendering motion coefficient video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            w2c = batch["w2cs"][0]
            # (3, 3).
            K = batch["Ks"][0]
            # (3, 3).
            img = batch["imgs"][0]
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, colors_override=motion_coef_colors
            )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "motion_coefs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )



    @torch.no_grad()
    def save_int_videos(self, epoch: int, w2c):
        if self.train_loader is None:
            return
        video_dir = osp.join(self.save_dir, "int_videos")
        os.makedirs(video_dir, exist_ok=True)

        video_dir = osp.join(self.save_dir, "int_videos", f"epoch_{epoch:04d}")
        os.makedirs(video_dir, exist_ok=True)

        print(video_dir, 'saving dict toooooo')
        fps = 15.0
        # Render video.
        video = []
        ref_pred_depths = []
        masks = []
        depth_min, depth_max = 1e6, 0
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc="Rendering video", leave=False)
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            K = batch["Ks"][0]
            # (H, W, 3).
            img = batch["imgs"][0]

            w2c = torch.tensor(w2c).float().to(img.device)
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, return_depth=True, return_mask=True
            )
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
            ref_pred_depth = rendered["depth"][0].cpu()
            ref_pred_depths.append(ref_pred_depth)
            depth_min = min(depth_min, ref_pred_depth.min().item())
            depth_max = max(depth_max, ref_pred_depth.quantile(0.99).item())
            if rendered["mask"] is not None:
                masks.append(rendered["mask"][0].cpu().squeeze(-1))

        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "rgbs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        # depth video
        depth_video = torch.stack(
            [
                apply_depth_colormap(
                    ref_pred_depth, near_plane=depth_min, far_plane=depth_max
                )
                for ref_pred_depth in ref_pred_depths
            ],
            dim=0,
        )
        iio.mimwrite(
            osp.join(video_dir, "depths.mp4"),
            make_video_divisble((depth_video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
        if len(masks) > 0:
            # mask video
            mask_video = torch.stack(masks, dim=0)
            iio.mimwrite(
                osp.join(video_dir, "masks.mp4"),
                make_video_divisble((mask_video.numpy() * 255).astype(np.uint8)),
                fps=fps,
            )

        '''# Render 2D track video.
        tracks_2d, target_imgs = [], []
        sample_interval = 10
        batch0 = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in self.train_loader.dataset[0].items()
        }
        # ().
        t = batch0["ts"]
        # (4, 4).
        # (3, 3).
        K = batch0["Ks"]
        # (H, W, 3).
        img = batch0["imgs"]
        # (H, W).
        bool_mask = batch0["masks"] > 0.5
        img_wh = img.shape[-2::-1]
        for batch in tqdm(
            self.train_loader, desc="Rendering 2D track video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            # (1, H, W, 3).
            target_imgs.append(batch["imgs"].cpu())
            # (1,).
            target_ts = batch["ts"]
            # (1, 4, 4).
            target_w2cs = w2c
            # (1, 3, 3).
            target_Ks = batch["Ks"]
            rendered = self.model.render(
                t,
                w2c[None],
                K[None],
                img_wh,
                target_ts=target_ts,
                target_w2cs=target_w2cs,
            )
            pred_tracks_3d = rendered["tracks_3d"][0][
                ::sample_interval, ::sample_interval
            ][bool_mask[::sample_interval, ::sample_interval]].swapaxes(0, 1)
            pred_tracks_2d = torch.einsum("bij,bpj->bpi", target_Ks, pred_tracks_3d)
            pred_tracks_2d = pred_tracks_2d[..., :2] / torch.clamp(
                pred_tracks_2d[..., 2:], min=1e-6
            )
            tracks_2d.append(pred_tracks_2d.cpu())
        tracks_2d = torch.cat(tracks_2d, dim=0)
        target_imgs = torch.cat(target_imgs, dim=0)
        track_2d_video = plot_correspondences(
            target_imgs.numpy(),
            tracks_2d.numpy(),
            query_id=cast(int, t),
        )
        iio.mimwrite(
            osp.join(video_dir, "tracks_2d.mp4"),
            make_video_divisble(np.stack(track_2d_video, 0)),
            fps=fps,
        )'''
        # Render motion coefficient video.
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            motion_coef_colors = torch.pca_lowrank(
                self.model.fg.get_coefs()[None],
                q=3,
            )[0][0]
        motion_coef_colors = (motion_coef_colors - motion_coef_colors.min(0)[0]) / (
            motion_coef_colors.max(0)[0] - motion_coef_colors.min(0)[0]
        )

        if self.model.bg is None: 
          motion_coef_colors = motion_coef_colors
        else:
          motion_coef_colors = F.pad(
              motion_coef_colors, (0, 0, 0, self.model.bg.num_gaussians), value=0.5
          )
        video = []
        for batch in tqdm(
            self.train_loader, desc="Rendering motion coefficient video", leave=False
        ):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            # ().
            t = batch["ts"][0]
            # (4, 4).
            # (3, 3).
            K = batch["Ks"][0]
            # (3, 3).
            img = batch["imgs"][0]
            img_wh = img.shape[-2::-1]
            rendered = self.model.render(
                t, w2c[None], K[None], img_wh, colors_override=motion_coef_colors
            )
            # Putting results onto CPU since it will consume unnecessarily
            # large GPU memory for long sequence OW.
            video.append(torch.cat([img, rendered["img"][0]], dim=1).cpu())
        video = torch.stack(video, dim=0)
        iio.mimwrite(
            osp.join(video_dir, "motion_coefs.mp4"),
            make_video_divisble((video.numpy() * 255).astype(np.uint8)),
            fps=fps,
        )
