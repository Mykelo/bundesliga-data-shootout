from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.processing import read_video_to_numpy


class DFLDataset(Dataset):
    videos_data: pd.DataFrame
    labels_path: str
    videos_dir: str
    label_map: dict[str, int]
    video_transform = None
    label_transform = None
    size: int | None
    random_state: int | None

    def __init__(
        self,
        labels_path: str,
        videos_dir: str,
        video_transform=None,
        label_transform=None,
        size: int | None = None,
        random_state: int | None = None,
    ):
        self.videos_data = pd.read_csv(labels_path)
        if size is not None:
            self.videos_data = self.videos_data.sample(
                n=size, random_state=random_state
            )

        self.labels_path = labels_path
        self.videos_dir = videos_dir
        self.video_transform = video_transform
        self.label_transform = label_transform
        self.label_map = {"nothing": 0, "challenge": 1, "throwin": 2, "play": 3}
        self.size = size
        self.random_state = random_state

    def __len__(self):
        return len(self.videos_data)

    def __getitem__(self, index) -> tuple[np.ndarray | torch.Tensor, int]:
        label_row = self.videos_data.iloc[index]
        video: np.ndarray | torch.Tensor = read_video_to_numpy(
            Path(self.videos_dir, f"{label_row['clip_id']}.mp4")
        )

        if self.video_transform is not None:
            video = self.video_transform(video)

        return video, self.label_map[label_row["event"]]
