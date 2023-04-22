from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.processing import read_video_to_numpy


class DFLDataset(Dataset):
    videos_data: pd.DataFrame
    data_dir: str
    label_map: dict[str, int]
    videos_to_include: list[str] | None
    clips_to_include: list[str] | None
    video_transform = None
    label_transform = None
    size: int | None
    random_state: int | None

    def __init__(
        self,
        data_dir: str,
        videos_to_include: list[str] | None = None,
        clips_to_include: list[str] | None = None,
        video_transform=None,
        label_transform=None,
        size: int | None = None,
        random_state: int | None = None,
    ):
        self.videos_data = pd.read_csv(Path(data_dir, "labels.csv"))
        if videos_to_include is not None:
            self.videos_data = self.videos_data[
                self.videos_data["video_id"].isin(videos_to_include)
            ]
        if clips_to_include is not None:
            self.videos_data = self.videos_data[
                self.videos_data["clip_id"].isin(clips_to_include)
            ]
        if size is not None:
            self.videos_data = self.videos_data.sample(
                n=size, random_state=random_state
            )

        self.data_dir = data_dir
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
            Path(self.data_dir, f"{label_row['clip_id']}.mp4")
        )

        if self.video_transform is not None:
            video = self.video_transform(video)

        return video, self.label_map[label_row["event"]]


def train_test_split(
    dataset: DFLDataset, test_size: float | int, random_state: int | None = None
) -> tuple[DFLDataset, DFLDataset]:
    video_ids = dataset.videos_data["video_id"].unique()
    if test_size < 1:
        test_size = int(test_size * len(video_ids))

    indices = np.array(range(len(video_ids)))
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(indices)
    test_indices = indices[: int(test_size)]
    train_indices = indices[int(test_size) :]

    test_videos, train_videos = video_ids[test_indices], video_ids[train_indices]
    df = dataset.videos_data
    test_df = df[df["video_id"].isin(test_videos)]
    train_df = df[df["video_id"].isin(train_videos)]

    test_dataset = DFLDataset(
        data_dir=dataset.data_dir,
        clips_to_include=test_df["clip_id"].to_list(),
        video_transform=dataset.video_transform,
        label_transform=dataset.label_transform,
    )
    train_dataset = DFLDataset(
        data_dir=dataset.data_dir,
        clips_to_include=train_df["clip_id"].to_list(),
        video_transform=dataset.video_transform,
        label_transform=dataset.label_transform,
    )

    return train_dataset, test_dataset
