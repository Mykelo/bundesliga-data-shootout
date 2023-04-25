import lightning.pytorch as pl
from src.data.datasets import DFLDataset
from src.data.transformations import VideoToTensor, VideoResize
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path


class DFLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        video_size: tuple[int, int] = (180, 320),
        skip_frames: int | None = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.video_size = video_size
        self.skip_frames = skip_frames

    def setup(self, stage: str):
        self.transformations = transforms.Compose(
            [VideoToTensor(), VideoResize(size=self.video_size)]
        )

        self.dfl_train = DFLDataset(
            labels_path=str(Path(self.data_dir, "train_labels.csv")),
            videos_dir=self.data_dir,
            video_transform=self.transformations,
            skip_frames=self.skip_frames,
        )
        self.dfl_test = DFLDataset(
            labels_path=str(Path(self.data_dir, "test_labels.csv")),
            videos_dir=self.data_dir,
            video_transform=self.transformations,
            skip_frames=self.skip_frames,
        )
        self.dfl_val = DFLDataset(
            labels_path=str(Path(self.data_dir, "test_labels.csv")),
            videos_dir=self.data_dir,
            video_transform=self.transformations,
            skip_frames=self.skip_frames,
        )

    def train_dataloader(self):
        return DataLoader(self.dfl_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dfl_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dfl_test, batch_size=self.batch_size)
