import os
import time
from dataclasses import dataclass

import torch
import tyro
from loguru import logger as guru

from flow3d.renderer import Renderer

torch.set_float32_matmul_precision("high")
# 
# class SceneModel(nn.Module):
#    def __init__(
#        self,
#        Ks: Tensor,
#        w2cs: Tensor,
#        fg_params: GaussianParams,
#        motion_bases: MotionBases,
#        bg_params: GaussianParams | None = None,
#    ):

@dataclass
class RenderConfig:
    work_dir: str
    port: int = 8890


def main(cfg: RenderConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"results/{cfg.work_dir}/checkpoints/last.ckpt"
    print(ckpt_path)

    #init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz'    
    #init_pt_path = '/data3/zihanwa3/Capstone-DSR/Appendix/Depth-Anything-V2/da_pt_cld_4_Stat.npz'
    init_pt_path='/data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz'  
    # /data3/zihanwa3/Capstone-DSR/Appendix/dust3r/patched_stat_imgs/pc.npz
    assert os.path.exists(ckpt_path)
    renderer = Renderer.init_from_pc_checkpoint(
        pc_dir=init_pt_path,
        device=device,
        work_dir=cfg.work_dir,
        port=cfg.port,
    )
    guru.info(f"Starting rendering from {renderer.global_step=}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))


    '''@staticmethod
    def init_from_checkpoint(
        path: str, device: torch.device, *args, **kwargs
    ) -> "Renderer":

        model = SceneModel(Ks, w2cs, fg, motion_bases, bg)
        model = model.to(device)
        renderer = Renderer(model, device, *args, **kwargs)
        renderer.global_step = 100
        renderer.epoch = 0
        return renderer

    @staticmethod
    def init_from_state_dict(state_dict, prefix=""):
        fg = GaussianParams.init_from_state_dict(
            state_dict, prefix=f"{prefix}fg.params."
        )
        bg = None
        if any("bg." in k for k in state_dict):
            bg = GaussianParams.init_from_state_dict(
                state_dict, prefix=f"{prefix}bg.params."
            )
        motion_bases = MotionBases.init_from_state_dict(
            state_dict, prefix=f"{prefix}motion_bases.params."
        )
        Ks = state_dict[f"{prefix}Ks"]
        w2cs = state_dict[f"{prefix}w2cs"]
        return 

    '''