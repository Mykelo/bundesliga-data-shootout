import pytorch_lightning as pl
from src.data.datasets import DFLDataset, train_test_split
from src.data.transformations import VideoToTensor, VideoResize
from torchvision import transforms
from torch.utils.data import DataLoader


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int = 32, random_state: int | None = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_state = random_state

    def setup(self, stage: str):
        self.transformations = transforms.Compose(
            [VideoToTensor(), VideoResize(size=(270, 480))]
        )
        self.dfl_full = DFLDataset(
            data_dir=self.data_dir,
            video_transform=self.transformations,
            label_transform=transforms.ToTensor(),
        )
        self.dfl_train, self.dfl_test = train_test_split(
            dataset=self.dfl_full, test_size=2, random_state=self.random_state
        )
        self.dfl_val = self.dfl_test

    def train_dataloader(self):
        return DataLoader(self.dfl_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dfl_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dfl_test, batch_size=self.batch_size)
