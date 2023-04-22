import typer
import logging
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from src.training.pl_modules import LitDFL
from src.data.data_modules import DFLDataModule
import torch


def main():
    """Trains the ResNet 3D model."""
    AVAIL_GPUS = min(1, torch.cuda.device_count())

    mlf_logger = MLFlowLogger(experiment_name="res_net_3d")

    model = LitDFL()
    dm = DFLDataModule(
        data_dir=str(Path("data", "processed")), random_state=42, batch_size=2
    )
    trainer = pl.Trainer(
        default_root_dir=Path("models", "r3d"),
        logger=mlf_logger,
        accelerator="cuda" if AVAIL_GPUS > 0 else "cpu",
        devices=AVAIL_GPUS,
        max_epochs=5,
        limit_train_batches=10,
        limit_val_batches=10,
        log_every_n_steps=1
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    typer.run(main)
