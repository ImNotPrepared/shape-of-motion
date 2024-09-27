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
    base_visuals_path =f'{cfg.work_dir}/visuals'
    os.makedirs(base_visuals_path, exist_ok=True)


    train_list = []
    for cfg in cfgs:
      train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
          get_train_val_datasets(cfg.data, load_val=True)
      )
      guru.info(f"Training dataset has {train_dataset.num_frames} frames")
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # save config
      os.makedirs(cfg.work_dir, exist_ok=True)
      with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
          yaml.dump(asdict(cfg), f, default_flow_style=False)

      # if checkpoint exists
      ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
      train_loader = DataLoader(
          train_dataset,
          batch_size=cfg.batch_size,
          num_workers=cfg.num_dl_workers,
          persistent_workers=True,
          collate_fn=BaseDataset.train_collate_fn,
      )

      train_list.append((train_loader , train_loader, train_dataset))
    train_dataset_0 = train_list[0][-1]#.train_step(batch_0)
    train_dataset_1 = train_list[1][-1]#.train_step(batch_1)
    train_dataset_2 = train_list[2][-1]#.train_step(batch_2)
    train_dataset_3 = train_list[3][-1]#.train_step(batch_3)


    train_loader_0 = train_list[0][1]
    train_loader_1 = train_list[1][1]
    train_loader_2 = train_list[2][1]
    train_loader_3 = train_list[3][1]

    train_loaders = [train_loader_0, train_loader_1, train_loader_2, train_loader_3]

    # Zip the loaders to load one batch from each loader at each step
    for batch_0, batch_1, batch_2, batch_3 in zip(*train_loaders):

        #### load multi-view data
        batch_0 = to_device(batch_0, device)
        batch_1 = to_device(batch_1, device)
        batch_2 = to_device(batch_2, device)
        batch_3 = to_device(batch_3, device)
        batch = [batch_0, batch_0, batch_0, batch_0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]

        rendered = renderer.model.render(
            ts[i].item(),
            w2cs[None, i],
            Ks[None, i],
            img_wh,
        )


    for cam_index in range(1,5):
        h, w = md['hw'][cam_index]
        def_pix = torch.tensor(
            np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
        pix_ones = torch.ones(h * w, 1).cuda().float()
        image_size, radius = (h, w), 0.01
        RENDER_MODE = 'color'
        print( np.array(json_file['k']).shape)
        w2c, k = (np.array((json_file['w2c'])[0][cam_index]), np.array(json_file['k'][0][cam_index]))
        w2c = np.linalg.inv(w2c)

        images = []
        #294-111=183
        reversed_range = new_range =list(range(111, -1, -5))

        for i, frame_index in enumerate(reversed_range):
            w2c = torch.tensor(w2c).float().cuda()

            scale_x=scale_y=1.0
            K_scaled = k.copy()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            #K_scaled[0, 2] *= scale_x  # cx
            #K_scaled[1, 2] *= scale_y  # cy
            K = torch.tensor(K_scaled).float().cuda()
            print(K.shape, w2c.shape)

            renderer.model.training = False

            img_wh = w, h
            t = torch.tensor(
              [frame_index]
            )
            im = renderer.model.render(t, w2c[None], K[None], img_wh)["img"][0]








            im=im.clip(0,1)
            print(im.shape, im.max())
            im = np.array(im.detach().cpu().numpy()) * 255


            new_width, new_height = w, h  # desired dimensions
            im = cv2.resize(im, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
            image = Image.fromarray((im).astype(np.uint8))
            images.append(np.array(image))
        print(f'saving to {(base_visuals_path)}')
        imageio.mimsave(os.path.join(base_visuals_path, f'cam_{cam_index}.gif'), images, fps=5)


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
