import typer
import logging
from typing import Optional, Tuple
from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.training.pl_modules import LitDFL
from src.data.data_modules import DFLDataModule
import torch


def main(
    experiment_name: str = "res_net_3d",
    data_dir: Path = typer.Option(exists=True, default=Path("data", "processed")),
    checkpoint_dir: Path = typer.Option(
        exists=True, dir_okay=True, default=Path("models")
    ),
    max_epochs: int = typer.Option(default=20),
    random_state: Optional[int] = typer.Option(default=42),
    video_size: Tuple[int, int] = typer.Option(default=(180, 320)),
    batch_size: int = typer.Option(default=2),
):
    """Trains the ResNet 3D model."""
    AVAIL_GPUS = min(1, torch.cuda.device_count())

    logger = logging.getLogger("Training ResNet 3D model")
    logger.info(f"Using {AVAIL_GPUS} GPUs")
    logger.info(f"experiment_name = {experiment_name}")
    logger.info(f"data_dir = {data_dir}")
    logger.info(f"checkpoint_dir = {checkpoint_dir}")
    logger.info(f"random_state = {random_state}")
    logger.info(f"video_size = {video_size}")
    logger.info(f"batch_size = {batch_size}")

    mlf_logger = MLFlowLogger(experiment_name=experiment_name)
    mlf_logger.log_hyperparams({"video_size": video_size, "batch_size": batch_size})

    model = LitDFL()
    dm = DFLDataModule(
        data_dir=str(data_dir),
        random_state=random_state,
        batch_size=batch_size,
        video_size=video_size,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        dirpath=Path(checkpoint_dir, experiment_name),
        filename=f"{experiment_name}_{mlf_logger.run_id}",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(
        default_root_dir=Path(checkpoint_dir, experiment_name),
        logger=mlf_logger,
        accelerator="cuda" if AVAIL_GPUS > 0 else "cpu",
        devices=AVAIL_GPUS,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        log_every_n_steps=1,
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    typer.run(main)
