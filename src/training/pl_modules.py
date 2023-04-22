import lightning.pytorch as pl
import torch.nn.functional as F
import torch
from src.models.models import R3DDFL
from sklearn.metrics import f1_score


class LitDFL(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = R3DDFL()
        self.training_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output_data = self._shared_eval_step(batch, batch_idx, "train")
        self.training_step_outputs.append(output_data)

        return output_data["loss"]

    def test_step(self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int):
        outputs = self._shared_eval_step(batch, batch_idx, "test")
        self.test_step_outputs.append(outputs)

    def validation_step(self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int):
        outputs = self._shared_eval_step(batch, batch_idx, "val")
        self.validation_step_outputs.append(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self) -> None:
        f1 = self._calc_f1(self.training_step_outputs)
        self.training_step_outputs.clear()
        self.logger.log_metrics({f"train_f1": f1})

    def on_validation_epoch_end(self) -> None:
        f1 = self._calc_f1(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        self.logger.log_metrics({f"val_f1": f1})

    def on_test_epoch_end(self) -> None:
        f1 = self._calc_f1(self.test_step_outputs)
        self.test_step_outputs.clear()
        self.logger.log_metrics({f"test_f1": f1})

    def _shared_eval_step(
        self, batch: list[torch.Tensor, torch.Tensor], batch_idx: int, step: str
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.logger.log_metrics(metrics={f"{step}_loss": loss.item()})

        return {"loss": loss, "pred": y_hat, "true": y}

    def _calc_f1(self, step_outputs: list[dict[str, torch.Tensor]]) -> float:
        true = torch.concat([step["true"] for step in step_outputs])
        pred = torch.concat([step["pred"] for step in step_outputs])
        pred = pred.argmax(dim=1)

        return f1_score(true.cpu(), pred.cpu(), labels=[0, 1, 2, 3], average="macro")
