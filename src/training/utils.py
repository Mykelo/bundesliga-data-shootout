from lightning.pytorch.loggers import MLFlowLogger
import lightning.pytorch as pl
from pathlib import Path
from src.data.data_modules import DFLDataModule
from lightning.pytorch.callbacks import ModelCheckpoint


def train_classification_model(
    module: pl.LightningModule,
    logger: MLFlowLogger,
    data_dir: Path,
    checkpoint_dir: Path,
    experiment_name: str,
    batch_size: int,
    video_size: tuple[int, int],
    max_epochs: int,
    gpus: int,
    skip_frames: int | None = None,
):
    dm = DFLDataModule(
        data_dir=str(data_dir),
        batch_size=batch_size,
        video_size=video_size,
        skip_frames=skip_frames,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        dirpath=Path(checkpoint_dir, experiment_name),
        filename=f"{experiment_name}_{logger.run_id}",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(
        default_root_dir=Path(checkpoint_dir, experiment_name),
        logger=logger,
        accelerator="cuda" if gpus > 0 else "cpu",
        devices=gpus,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        log_every_n_steps=1,
    )

    trainer.fit(model=module, datamodule=dm)
