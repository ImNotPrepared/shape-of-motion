from dataclasses import asdict, replace

from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .casual_dataset import CasualDataset, CustomDataConfig, CasualDatasetVideoView, DavisDataConfig
from .iphone_dataset import (
    iPhoneDataConfig,
    iPhoneDataset,
    iPhoneDatasetKeypointView,
    iPhoneDatasetVideoView,
)
import numpy as np 


class SynchornizedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lennnn = len(self.datasets[0])
        print(len(self.datasets))
        self.num_targets_per_frame = self.datasets[0].num_targets_per_frame

    def __len__(self):
        return self.lennnn 
    

    def __getitem__(self, index: int):
        # Generate synchronized random indices for target frames
        index = np.random.randint(0, self.lennnn)

        target_inds = np.random.choice(
                self.lennnn, (self.num_targets_per_frame,), replace=False
            )
        data_list = []
        for dataset in self.datasets:
            data = dataset.__getitem__(index, target_inds=target_inds)
            data_list.append(data)

        return data_list
    


def get_train_val_datasets(
    data_cfg: iPhoneDataConfig | DavisDataConfig | CustomDataConfig, load_val: bool
) -> tuple[BaseDataset, Dataset | None, Dataset | None, Dataset | None]:
    train_video_view = None
    val_img_dataset = None
    val_kpt_dataset = None
    if isinstance(data_cfg, iPhoneDataConfig):
        train_dataset = iPhoneDataset(**asdict(data_cfg))
        train_video_view = iPhoneDatasetVideoView(train_dataset)
        if load_val:
            val_img_dataset = (
                iPhoneDataset(
                    **asdict(replace(data_cfg, split="val", load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
            val_kpt_dataset = iPhoneDatasetKeypointView(train_dataset)
    elif isinstance(data_cfg, DavisDataConfig) or isinstance(
        data_cfg, CustomDataConfig
    ):
        data_cfg_dict = asdict(data_cfg)
        train_dataset = CasualDataset(**data_cfg_dict)
        train_video_view = CasualDatasetVideoView(train_dataset)


    else:
        raise ValueError(f"Unknown data config: {data_cfg}")


    return train_dataset, train_video_view, val_img_dataset, val_kpt_dataset
