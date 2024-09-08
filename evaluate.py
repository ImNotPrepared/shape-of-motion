import os
import time
from dataclasses import dataclass
import os
import time
from dataclasses import dataclass
import json
import torch
import tyro
import numpy as np
from loguru import logger as guru
import imageio
from flow3d.renderer import Renderer
import torch
import tyro
from loguru import logger as guru
import cv2
from flow3d.renderer import Renderer
from PIL import Image

torch.set_float32_matmul_precision("high")


@dataclass
class RenderConfig:
    work_dir: str


def main(cfg: RenderConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    assert os.path.exists(ckpt_path)

    renderer = Renderer.init_from_checkpoint(
        ckpt_path,
        device,
        work_dir=cfg.work_dir,
        port=None,
    )
    base_data_path = '/data3/zihanwa3/Capstone-DSR/Dynamic3DGaussians/data_ego'
    seq='cmu_bike'

    file_path = os.path.join(base_data_path, seq, 'Dy_train_meta.json') 
    with open(file_path, 'r') as file:
        json_file = json.load(file)
    md=json_file
    base_visuals_path ='./visuals'
    os.makedirs(base_visuals_path, exist_ok=True)
    for cam_index in range(1,5):
        h, w = md['hw'][cam_index]
        def_pix = torch.tensor(
            np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
        pix_ones = torch.ones(h * w, 1).cuda().float()
        image_size, radius = (h, w), 0.01
        RENDER_MODE = 'color'
        w2c, k = (np.array((json_file['w2c'])[0][cam_index]), np.array(json_file['k'][0][cam_index]))
        w2c = np.linalg.inv(w2c)

        images = []
        #294-111=183
        reversed_range = list(range(111))
        

        for i, frame_index in enumerate(reversed_range):
            print(frame_index,  cam_index)
            w2c, K = (np.array((json_file['w2c'])[frame_index][cam_index]), np.array(json_file['k'][frame_index][cam_index]))
            #w2c = np.linalg.inv(w2c)
            w2c = torch.tensor(w2c).float().cuda()
            K = torch.tensor(K).float().cuda()
            print(K.shape, w2c.shape)

            renderer.model.training = False
            #w, h = img_wh

            img_wh = w, h
            t = torch.tensor(
              frame_index
            )
            im = renderer.model.render(t, w2c[None], K[None], img_wh)["img"][0]
            im=im.clip(0,1)
            print(im.shape, im.max())
            im = np.array(im.detach().cpu().numpy()) * 255

            # im = np.array(im.detach().cpu().permute(1, 2, 0).numpy()) * 255
            
            new_width, new_height = 256, 144  # desired dimensions
            im = cv2.resize(im, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray((im).astype(np.uint8))
            images.append(np.array(image))
        imageio.mimsave(os.path.join(base_visuals_path, f'cam_{cam_index}.gif'), images, fps=5)






if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
