from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.processing import read_video_to_numpy


class DFLDataset(Dataset):
    videos_data: pd.DataFrame
    annotations_file: str
    video_dir: str
    label_map: dict[str, int]
    video_transform = None
    label_transform = None

    def __init__(
        self,
        annotations_file: str,
        video_dir: str,
        videos_to_include: list[str] | None = None,
        video_transform=None,
        label_transform=None,
    ):
        self.annotations_file = annotations_file
        self.videos_data = pd.read_csv(annotations_file)
        if videos_to_include is not None:
            self.videos_data = self.videos_data[
                self.videos_data["video_id"].isin(videos_to_include)
            ]
        self.video_dir = video_dir
        self.video_transform = video_transform
        self.label_transform = label_transform
        self.label_map = {"nothing": 0, "challenge": 1, "throwin": 2, "play": 3}

    def __len__(self):
        return len(self.videos_data)

    def __getitem__(self, index) -> tuple[np.ndarray, int]:
        label_row = self.videos_data.iloc[index]
        video = read_video_to_numpy(Path(self.video_dir, f"{label_row['clip_id']}.mp4"))

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
    test_indices = indices[:int(test_size)]
    train_indices = indices[int(test_size):]

    test_videos, train_videos = video_ids[test_indices], video_ids[train_indices]
    test_dataset = DFLDataset(
        annotations_file=dataset.annotations_file,
        video_dir=dataset.video_dir,
        videos_to_include=test_videos,
        video_transform=dataset.video_transform,
        label_transform=dataset.label_transform,
    )
    train_dataset = DFLDataset(
        annotations_file=dataset.annotations_file,
        video_dir=dataset.video_dir,
        videos_to_include=train_videos,
        video_transform=dataset.video_transform,
        label_transform=dataset.label_transform,
    )

    return train_dataset, test_dataset
