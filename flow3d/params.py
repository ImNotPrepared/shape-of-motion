import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.transforms import cont_6d_to_rmat


class GaussianParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        motion_coefs: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
        scene_scale: torch.Tensor | float = 1.0,
        feats: torch.Tensor | None = None,
    ):
        super().__init__()

        params_dict = {
            "means": nn.Parameter(means),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(scales),
            "colors": nn.Parameter(colors),
            "opacities": nn.Parameter(opacities),
        }
        if motion_coefs is not None:
            params_dict["motion_coefs"] = nn.Parameter(motion_coefs)
        if feats is not None:
            params_dict["feats"] = nn.Parameter(feats)
        self.params = nn.ParameterDict(params_dict)
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)

        if scene_center is None:
            scene_center = torch.zeros(3, device=means.device)
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_scale", torch.as_tensor(scene_scale))

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        req_keys = ["means", "quats", "scales", "colors", "opacities"]
        print(state_dict.keys())
        assert all(f"{prefix}{k}" in state_dict for k in req_keys)
        args = {
            "motion_coefs": None,
            "feats": None, 
            "scene_center": torch.zeros(3),
            "scene_scale": torch.tensor(1.0),
        }
        for k in req_keys + list(args.keys()):
            if f"{prefix}{k}" in state_dict:
                args[k] = state_dict[f"{prefix}{k}"]
        return GaussianParams(**args)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def get_colors(self) -> torch.Tensor:
        return self.color_activation(self.params["colors"])

    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.params["scales"])

    def get_opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.params["opacities"])

    def get_quats(self) -> torch.Tensor:
        return self.quat_activation(self.params["quats"])

    def get_coefs(self) -> torch.Tensor:
        assert "motion_coefs" in self.params
        return self.motion_coef_activation(self.params["motion_coefs"])

    def get_feats(self) -> torch.Tensor:
        #assert "feats" in self.params
        try:
          return (self.params["feats"]) #self.motion_coef_activation
        except:
          print('NOT FINDING ANY FEAT')
          return torch.randn(self.get_colors().shape[0], 32).cuda()

    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_split = x[should_split].repeat([2] + [1] * (x.ndim - 1))
            if name == "scales":
                x_split -= math.log(1.6)
            x_new = nn.Parameter(torch.cat([x[~should_split], x_dup, x_split], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def reset_opacities(self, new_val):
        """
        reset all opacities to new_val
        """
        self.params["opacities"].data.fill_(new_val)
        updated_params = {"opacities": self.params["opacities"]}
        return updated_params


class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]
        assert check_bases_sizes(rots, transls)
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
            }
        )

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["rots", "transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return MotionBases(**args)

    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        transls = self.params["transls"][:, ts]  # (K, B, 3)
        rots = self.params["rots"][:, ts]  # (K, B, 6)
        transls = torch.einsum("pk,kni->pni", coefs.float(), transls.float())
        rots = torch.einsum("pk,kni->pni", coefs.float(), rots.float())  # (G, B, 6)
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        return torch.cat([rotmats, transls[..., None]], dim=-1)

class GaussianParamsOthers(GaussianParams):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation =  lambda x: x# torch.clamp(x, 0, 1)
# lambda x: torch.clamp(x, 0, 1)
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid


def check_gaussian_sizes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    motion_coefs: torch.Tensor | None = None,
) -> bool:
    dims = means.shape[:-1]
    print('*-*-*-'*70, means.shape, quats.shape, scales.shape, colors.shape, opacities.shape)

    if motion_coefs is not None:
      print(f'(MOTION_COEFS{motion_coefs.shape})')
    leading_dims_match = (
        quats.shape[:-1] == dims
        and scales.shape[:-1] == dims
        and colors.shape[:-1] == dims
        and opacities.shape == dims
    )
    if motion_coefs is not None and motion_coefs.numel() > 0:
        leading_dims_match &= motion_coefs.shape[:-1] == dims
    dims_correct = (
        means.shape[-1] == 3
        and (quats.shape[-1] == 4)
        and (scales.shape[-1] == 3)
        and (colors.shape[-1] == 3)
    )
    return leading_dims_match and dims_correct


def check_bases_sizes(motion_rots: torch.Tensor, motion_transls: torch.Tensor) -> bool:
    return (
        motion_rots.shape[-1] == 6
        and motion_transls.shape[-1] == 3
        and motion_rots.shape[:-2] == motion_transls.shape[:-2]
    )
