from dataclasses import dataclass


@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 0 #1e-2
    feats: float = 1e-3
    motion_coefs: float = 1e-3


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    feats: float = 1e-3


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4


@dataclass
class SceneLRConfig:
    fg: FGLRConfig
    bg: BGLRConfig
    motion_bases: MotionLRConfig

@dataclass
class LossesConfig:
    w_rgb: float = 7.0
    w_feat: float = 7.0 #0.01
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.1
    w_depth_grad: float = 1
    w_track: float = 4.0
    w_mask: float = 7.0
    w_smooth_bases: float = 0.1
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0


@dataclass
class LossesGTConfig:
    w_rgb: float = 1.0
    w_feat: float = 1.0 #0.01
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.1
    w_depth_grad: float = 1
    w_track: float = 2.0
    w_mask: float = 1.0
    w_smooth_bases: float = 0.1
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0


@dataclass
class OptimizerConfig:
    max_steps: int = 5000
    ## Adaptive gaussian control
    warmup_steps: int = 200
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002# 0.0002 # 0.0002
    densify_scale_threshold: float = 0.01 
          #  should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
      #  should_dup = is_grad_too_high & ~is_scale_too_big
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    ### Cull.
    cull_opacity_threshold: float = 0.1# 0.1
    # is_opacity_too_small = opacities < cfg.cull_opacity_threshold
    cull_scale_threshold: float = 0.5# 0.5
    # is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
    cull_screen_threshold: float = 0.15#0.15
    ##                is_radius_too_big = (
    #                self.running_stats["max_radii"] > cfg.cull_screen_threshold
    #            )
    #feat: float = 0.01
