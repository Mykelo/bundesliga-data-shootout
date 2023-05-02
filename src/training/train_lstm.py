import typer
import logging
from typing import Tuple
from pathlib import Path
from lightning.pytorch.loggers import MLFlowLogger
from src.training.lightning_modules import LitDFL
from src.models.models import ResNetLSTM
from src.training.utils import train_classification_model
import torch


def main(
    experiment_name: str = "res_net_lstm",
    data_dir: Path = typer.Option(exists=True, default=Path("data", "processed")),
    checkpoint_dir: Path = typer.Option(
        exists=True, dir_okay=True, default=Path("models")
    ),
    max_epochs: int = typer.Option(default=20),
    video_size: Tuple[int, int] = typer.Option(default=(180, 320)),
    batch_size: int = typer.Option(default=2),
    skip_frames: int = typer.Option(default=1),
    lr: float = 1e-3,
):
    """Trains the ResNet+LSTM model."""
    AVAIL_GPUS = min(1, torch.cuda.device_count())

    logger = logging.getLogger("Training ResNet+LSTM model")
    logger.info(f"Using {AVAIL_GPUS} GPUs")
    logger.info(f"experiment_name = {experiment_name}")
    logger.info(f"data_dir = {data_dir}")
    logger.info(f"checkpoint_dir = {checkpoint_dir}")
    logger.info(f"video_size = {video_size}")
    logger.info(f"batch_size = {batch_size}")
    logger.info(f"learning_rate = {lr}")
    logger.info(f"skip_frames = {skip_frames}")

    mlf_logger = MLFlowLogger(experiment_name=experiment_name)
    mlf_logger.log_hyperparams(
        {
            "video_size": video_size,
            "batch_size": batch_size,
            "learning_rate": lr,
            "skip_frames": skip_frames,
        }
    )

    module = LitDFL(model=ResNetLSTM(), learning_rate=lr)
    train_classification_model(
        module=module,
        logger=mlf_logger,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        batch_size=batch_size,
        video_size=video_size,
        max_epochs=max_epochs,
        gpus=AVAIL_GPUS,
        skip_frames=skip_frames,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    typer.run(main)
