import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import torch
from torchvision import models


class LitDFL(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.r3d = models.video.r3d_18(
            weights=models.video.R3D_18_Weights.KINETICS400_V1
        )
        self.r3d.fc = nn.Linear(in_features=512, out_features=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.r3d(x.permute(0, 2, 1, 3, 4))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx: int):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss)

    def _shared_eval_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        x = x.permute(0, 2, 1, 3, 4)
        y_hat = self.r3d(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
