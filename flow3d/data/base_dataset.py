from abc import abstractmethod

import torch
from torch.utils.data import Dataset, default_collate


class BaseDataset(Dataset):
    @property
    @abstractmethod
    def num_frames(self) -> int: ...

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return torch.arange(self.num_frames)

    @abstractmethod
    def get_w2cs(self) -> torch.Tensor: ...

    @abstractmethod
    def get_Ks(self) -> torch.Tensor: ...

    @abstractmethod
    def get_image(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_depth(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_mask(self, index: int) -> torch.Tensor: ...

    def get_img_wh(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_tracks_3d(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns 3D tracks:
            coordinates (N, T, 3),
            visibles (N, T),
            invisibles (N, T),
            confidences (N, T),
            colors (N, 3)
        """
        ...

    @abstractmethod
    def get_bkgd_points(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns background points:
            coordinates (N, 3),
            normals (N, 3),
            colors (N, 3)
        """
        ...

    @staticmethod
    def train_collate_fn(batch):
        collated = {}
        for k in batch[0]:
            if k not in [
                "query_tracks_2d",
                "target_ts",
                "target_w2cs",
                "target_Ks",
                "target_tracks_2d",
                "target_visibles",
                "target_track_depths",
                "target_invisibles",
                "target_confidences",
            ]:
                collated[k] = default_collate([sample[k] for sample in batch])
            else:
                collated[k] = [sample[k] for sample in batch]
        return collated
    
    @staticmethod
    def train_collate_fn_sync(batch):
        """
        Custom collate function to handle batches from SynchornizedDataset.
        
        Args:
            batch: List of samples, where each sample is a list of data dictionaries from each dataset.
            
        Returns:
            batches: List of collated batches, one for each dataset.
        """
        # Transpose the batch to group data by dataset
        # batch: List of samples -> batch_per_dataset: List of datasets
        # Each element in batch_per_dataset is a list of data dictionaries from all samples for that dataset
        batch_per_dataset = list(zip(*batch))
        
        batches = []
        for dataset_samples in batch_per_dataset:
            # dataset_samples: List of data dictionaries from samples for one dataset
            collated = {}
            for key in dataset_samples[0]:
                if key in [
                    "query_tracks_2d",
                    "target_ts",
                    "target_w2cs",
                    "target_Ks",
                    "target_tracks_2d",
                    "target_visibles",
                    "target_track_depths",
                    "target_invisibles",
                    "target_confidences",
                    "frame_names",
                ]:
                    # For list data, collect them into a list
                    collated[key] = [sample[key] for sample in dataset_samples]
                else:
                    # For tensor data, use default_collate
                    collated[key] = default_collate([sample[key] for sample in dataset_samples])
            batches.append(collated)
        return batches