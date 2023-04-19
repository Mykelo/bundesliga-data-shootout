import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from src.models.models import R3DDFL


class LitDFL(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = R3DDFL()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx: int):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log("test_loss", loss)

    def _shared_eval_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
