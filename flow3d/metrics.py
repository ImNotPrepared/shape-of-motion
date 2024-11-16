from typing import Literal

import numpy as np
import torch
import os
import torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE


def compute_psnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> float:
    """
    Args:
        preds (torch.Tensor): (..., 3) predicted images in [0, 1].
        targets (torch.Tensor): (..., 3) target images in [0, 1].
        masks (torch.Tensor | None): (...,) optional binary masks where the
            1-regions will be taken into account.

    Returns:
        psnr (float): Peak signal-to-noise ratio.
    """
    if masks is None:
        masks = torch.ones_like(preds[..., 0])
    return (
        -10.0
        * torch.log(
            F.mse_loss(
                preds * masks[..., None],
                targets * masks[..., None],
                reduction="sum",
            )
            / masks.sum().clamp(min=1.0)
            / 3.0
        )
        / np.log(10.0)
    ).item()


def compute_pose_errors(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float, float]:
    """
    Args:
        preds: (N, 4, 4) predicted camera poses.
        targets: (N, 4, 4) target camera poses.

    Returns:
        ate (float): Absolute trajectory error.
        rpe_t (float): Relative pose error in translation.
        rpe_r (float): Relative pose error in rotation (degree).
    """
    # Compute ATE.
    ate = torch.linalg.norm(preds[:, :3, -1] - targets[:, :3, -1], dim=-1).mean().item()
    # Compute RPE_t and RPE_r.
    # NOTE(hangg): It's important to use numpy here for the accuracy of RPE_r.
    # torch has numerical issues for acos when the value is close to 1.0, i.e.
    # RPE_r is supposed to be very small, and will result in artificially large
    # error.
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    pred_rels = np.linalg.inv(preds[:-1]) @ preds[1:]
    pred_rels = np.linalg.inv(preds[:-1]) @ preds[1:]
    target_rels = np.linalg.inv(targets[:-1]) @ targets[1:]
    error_rels = np.linalg.inv(target_rels) @ pred_rels
    traces = error_rels[:, :3, :3].trace(axis1=-2, axis2=-1)
    rpe_t = np.linalg.norm(error_rels[:, :3, -1], axis=-1).mean().item()
    rpe_r = (
        np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0)).mean().item()
        / np.pi
        * 180.0
    )
    return ate, rpe_t, rpe_r


class mPSNR(PeakSignalNoiseRatio):
    sum_squared_error: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(self, **kwargs) -> None:
        super().__init__(
            data_range=1.0,
            base=10.0,
            dim=None,
            reduction="elementwise_mean",
            **kwargs,
        )
        self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (..., 3) float32 predicted images.
            targets (torch.Tensor): (..., 3) float32 target images.
            masks (torch.Tensor | None): (...,) optional binary masks where the
                1-regions will be taken into account.
        """
        # torch.Size([1, 288, 512, 3]) torch.Size([288, 512, 3]) torch.Size([288, 512, 3])

        ### correct one: torch.Size([1, 288, 512, 3]) torch.Size([101, 288, 512, 3]) torch.Size([101, 288, 512])
        print(preds.shape, targets.shape, masks.shape)
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        self.sum_squared_error.append(
            torch.sum(torch.pow((preds - targets) * masks[..., None], 2))
        )
        self.total.append(masks.sum().to(torch.int64) * 3)

    def compute(self) -> torch.Tensor:
        """Compute peak signal-to-noise ratio over state."""
        sum_squared_error = dim_zero_cat(self.sum_squared_error)
        total = dim_zero_cat(self.total)
        return -10.0 * torch.log(sum_squared_error / total).mean() / np.log(10.0)

def unproject_image(img, w2c, K, depth, mask, glb_pc, img_wh, output_dir='output_depth_maps', output_filename='estimated_depth.png'):
    device = img.device
    w2c = w2c.to(device)
    K = K.to(device)
    depth = depth.to(device)
    mask = mask.to(device)
    pc = torch.from_numpy(glb_pc).float().to(device)  # (N, 3)
    ones = torch.ones((pc.shape[0], 1), device=device, dtype=pc.dtype)

    pc_hom = torch.cat([pc, ones], dim=1)  # (N, 4)
    c_pc_hom = pc_hom @ w2c.T  # (N, 4)
    c_pc = c_pc_hom[:, :3]  # (N, 3)
    x, y, z = c_pc[:, 0], c_pc[:, 1], c_pc[:, 2]  # (N,)

    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    # Stack coordinates for matrix multiplication
    coords = torch.stack([x, y, z], dim=0)  # (3, N_valid)
    uv = K @ coords
    u = uv[0, :] / uv[2, :]
    v = uv[1, :] / uv[2, :]

    # Round to nearest pixel indices
    u_pixel = torch.round(u).long()
    v_pixel = torch.round(v).long()

    # Unpack image dimensions
    W, H = img_wh

    # Filter points within image bounds
    in_bounds = (u_pixel >= 0) & (u_pixel < W) & (v_pixel >= 0) & (v_pixel < H)
    u_pixel = u_pixel[in_bounds]
    v_pixel = v_pixel[in_bounds]
    z = z[in_bounds]

    # Compute linear pixel indices
    pixel_indices = v_pixel * W + u_pixel  # (N_in_bounds,)

    # Initialize estimated depth map
    estimated_depth_flat = torch.full((H * W,), float('inf'), device=device, dtype=pc.dtype)

    # Manually assign depth values to the estimated depth map
    # Sort pixel_indices and corresponding z values
    sorted_indices = torch.argsort(pixel_indices)
    sorted_pixel_indices = pixel_indices[sorted_indices]
    sorted_z = z[sorted_indices]

    # Find the indices where pixel_indices change
    change_indices = torch.cat([
        torch.tensor([0], device=sorted_pixel_indices.device),
        (sorted_pixel_indices[1:] != sorted_pixel_indices[:-1]).nonzero(as_tuple=True)[0] + 1,
        torch.tensor([len(sorted_pixel_indices)], device=sorted_pixel_indices.device)
    ])

    # Loop over unique pixel indices to assign the minimum depth value
    for i in range(len(change_indices) - 1):
        start = change_indices[i].item()
        end = change_indices[i + 1].item()
        idx = sorted_pixel_indices[start].item()
        min_z = sorted_z[start:end].min()
        estimated_depth_flat[idx] = min_z

    # Reshape to (H, W)
    estimated_depth = estimated_depth_flat.view(H, W)

    gt_depth = depth
    valid_mask = (estimated_depth != float('inf')) & (gt_depth > 0) & (1-mask).bool()

    # Extract valid depth values
    est_depth_valid = estimated_depth[valid_mask]
    gt_depth_valid = gt_depth[valid_mask]

    # Calculate Absolute Relative Error
    abs_rel_error = torch.mean(torch.abs(gt_depth_valid - est_depth_valid) / gt_depth_valid)
    print(f"Absolute Relative Error: {abs_rel_error.item()}")

    # Save the depth map
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    save_depth_map(estimated_depth, output_path)

    return abs_rel_error

def save_depth_map(depth_map, filename):
    import matplotlib.pyplot as plt
    plt.imshow(depth_map.cpu(), cmap='plasma', vmin=0, vmax=depth_map[depth_map != float('inf')].max())
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def mask_iou(mask1, mask2):
    """
    Computes the Intersection over Union (IoU) of two binary masks using PyTorch tensors.

    Parameters:
    - mask1: torch.Tensor of shape (H, W), binary mask
    - mask2: torch.Tensor of shape (H, W), binary mask

    Returns:
    - IoU: torch.Tensor, Intersection over Union value between 0 and 1
    """
    # Ensure masks are boolean tensors
    mask1 = (mask1 > 0.9).bool()
    mask2 = (mask2 > 0.9).bool()

    # Compute intersection and union
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)

    # Sum over the pixels
    intersection_sum = torch.sum(intersection).float()
    union_sum = torch.sum(union).float()
    print('iou', intersection_sum, union_sum, intersection_sum / union_sum)
    if union_sum == 0:
        # Both masks are empty; define IoU as 1
        return torch.tensor(1.0)
    else:
        iou = intersection_sum / union_sum
        return iou

    

class mMDE(StructuralSimilarityIndexMeasure):
    similarity: list

    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=None,
            data_range=1.0,
            return_full_image=False,
            **kwargs,
        )
        assert isinstance(self.sigma, float)

    def __len__(self) -> int:
        return sum([s.shape[0] for s in self.similarity])

    def update(
        self,
        img, w2c, K, depth, mask, glb_pc, img_wh,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional binary masks where
                the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(img[..., 0])
        score = unproject_image(
                img, w2c, K, depth, mask, glb_pc, img_wh
        )

        self.similarity.append(score)

    def compute(self) -> torch.Tensor:
        """Compute final SSIM metric."""
        return torch.tensor(self.similarity).mean()


class mIOU(StructuralSimilarityIndexMeasure):
    similarity: list

    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=None,
            data_range=1.0,
            return_full_image=False,
            **kwargs,
        )
        assert isinstance(self.sigma, float)

    def __len__(self) -> int:
        return sum([s.shape[0] for s in self.similarity])

    def update(
        self,
        pred_mask, target_mask,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional binary masks where
                the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(pred_mask[..., 0])

        self.similarity.append(mask_iou(pred_mask, target_mask))

    def compute(self) -> torch.Tensor:
        """Compute final SSIM metric."""
        print(self.similarity)
        return torch.tensor(self.similarity).mean()
    

class mSSIM(StructuralSimilarityIndexMeasure):
    similarity: list

    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=None,
            data_range=1.0,
            return_full_image=False,
            **kwargs,
        )
        assert isinstance(self.sigma, float)

    def __len__(self) -> int:
        return sum([s.shape[0] for s in self.similarity])

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update state with predictions and targets.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional binary masks where
                the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])

        # Construct a 1D Gaussian blur filter.
        assert isinstance(self.kernel_size, int)
        hw = self.kernel_size // 2
        shift = (2 * hw - self.kernel_size + 1) / 2
        assert isinstance(self.sigma, float)
        f_i = (
            (torch.arange(self.kernel_size, device=preds.device) - hw + shift)
            / self.sigma
        ) ** 2
        filt = torch.exp(-0.5 * f_i)
        filt /= torch.sum(filt)

        # Blur in x and y (faster than the 2D convolution).
        def convolve2d(z, m, f):
            # z: (B, H, W, C), m: (B, H, W), f: (Hf, Wf).
            z = z.permute(0, 3, 1, 2)
            m = m[:, None]
            f = f[None, None].expand(z.shape[1], -1, -1, -1)
            z_ = torch.nn.functional.conv2d(
                z * m, f, padding="valid", groups=z.shape[1]
            )
            m_ = torch.nn.functional.conv2d(m, torch.ones_like(f[:1]), padding="valid")
            return torch.where(
                m_ != 0, z_ * torch.ones_like(f).sum() / (m_ * z.shape[1]), 0
            ).permute(0, 2, 3, 1), (m_ != 0)[:, 0].to(z.dtype)

        filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
        filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])
        filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

        mu0 = filt_fn(preds, masks)[0]
        mu1 = filt_fn(targets, masks)[0]
        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = filt_fn(preds**2, masks)[0] - mu00
        sigma11 = filt_fn(targets**2, masks)[0] - mu11
        sigma01 = filt_fn(preds * targets, masks)[0] - mu01

        # Clip the variances and covariances to valid values.
        # Variance must be non-negative:
        sigma00 = sigma00.clamp(min=0.0)
        sigma11 = sigma11.clamp(min=0.0)
        sigma01 = torch.sign(sigma01) * torch.minimum(
            torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
        )

        assert isinstance(self.data_range, float)
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
        denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
        ssim_map = numer / denom

        self.similarity.append(ssim_map.mean(dim=(1, 2, 3)))

    def compute(self) -> torch.Tensor:
        """Compute final SSIM metric."""
        return torch.cat(self.similarity).mean()


class mLPIPS(Metric):
    sum_scores: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that torchvision is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torchvision`."
            )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}."
            )
        self.net = _NoTrainLpips(net=net_type, spatial=True)

        self.add_state("sum_scores", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Update internal states with lpips scores.

        Args:
            preds (torch.Tensor): (B, H, W, 3) float32 predicted images.
            targets (torch.Tensor): (B, H, W, 3) float32 target images.
            masks (torch.Tensor | None): (B, H, W) optional float32 binary
                masks where the 1-regions will be taken into account.
        """
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        scores = self.net(
            (preds * masks[..., None]).permute(0, 3, 1, 2),
            (targets * masks[..., None]).permute(0, 3, 1, 2),
            normalize=True,
        )
        self.sum_scores.append((scores * masks[:, None]).sum())
        self.total.append(masks.sum().to(torch.int64))

    def compute(self) -> torch.Tensor:
        """Compute final perceptual similarity metric."""
        return (
            torch.tensor(self.sum_scores, device=self.device)
            / torch.tensor(self.total, device=self.device)
        ).mean()


class PCK(Metric):
    correct: list[torch.Tensor]
    total: list[int]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, threshold: float):
        """Update internal states with PCK scores.

        Args:
            preds (torch.Tensor): (N, 2) predicted 2D keypoints.
            targets (torch.Tensor): (N, 2) targets 2D keypoints.
            threshold (float): PCK threshold.
        """

        self.correct.append(
            (torch.linalg.norm(preds - targets, dim=-1) < threshold).sum()
        )
        self.total.append(preds.shape[0])

    def compute(self) -> torch.Tensor:
        """Compute PCK over state."""
        return (
            torch.tensor(self.correct, device=self.device)
            / torch.clamp(torch.tensor(self.total, device=self.device), min=1e-8)
        ).mean()
    

class MaskIoU(Metric):
    intersection: list[torch.Tensor]
    union: list[torch.Tensor]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("intersection", default=[], dist_reduce_fx="cat")
        self.add_state("union", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.union)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update state with predicted and target masks.

        Args:
            preds (torch.Tensor): Predicted binary mask (..., H, W), values in {0, 1}.
            targets (torch.Tensor): Target binary mask (..., H, W), values in {0, 1}.
        """
        preds = preds.to(torch.bool)
        targets = targets.to(torch.bool)
        self.intersection.append(torch.logical_and(preds, targets).float().sum())
        self.union.append(torch.logical_or(preds, targets).float().sum())

    def compute(self) -> torch.Tensor:
        """Compute final mask IoU over state."""
        intersection = dim_zero_cat(self.intersection)
        union = dim_zero_cat(self.union)
        return (intersection / union.clamp(min=1e-8)).mean()