from pathlib import Path
import typer
import logging
import pandas as pd
import numpy as np
import cv2
from src.data.processing import extract_frames, save_frames, sample_negative
from tqdm import tqdm
from typing import Tuple, Optional

app = typer.Typer()


@app.command(help="Extracts short clips from football matches in the given directory")
def extract(
    videos_dir: Path = typer.Option(exists=True, default=Path("data", "raw", "train")),
    labels_path: Path = typer.Option(
        exists=True, file_okay=True, default=Path("data", "raw", "train.csv")
    ),
    output_dir: Path = typer.Option(exists=True, default=Path("data", "processed")),
    negative_samples_num: int = typer.Option(default=4000),
    window_size: int = typer.Option(default=32),
    frame_size: Tuple[int, int] | None = typer.Option(default=None),
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger("Videos processing step")
    logger.info("Reading and extracting clips from match recordings...")
    logger.info(
        f"window_size = {window_size}, negative_samples_num = {negative_samples_num}"
    )
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df[~labels_df["event"].isin(["start", "end"])]
    labels_df = labels_df.drop(columns=["event_attributes"])

    labels_df["clip_id"] = ""

    all_videos = labels_df["video_id"].unique()
    neg_samples_per_video = negative_samples_num // len(all_videos)

    t = tqdm(total=len(labels_df) + negative_samples_num)

    video_ids: list[str] = []
    clip_ids: list[str] = []
    times: list[float] = []
    events: list[str] = []
    fps_values: list[int] = []
    clip_frames: list[int] = []
    frame_counts: list[int] = []
    for video_id in all_videos:
        path = videos_dir / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        df = labels_df[labels_df["video_id"] == video_id].reset_index(drop=True)
        center_frames = df["time"].to_numpy() * fps
        negative_samples_frames = sample_negative(
            neg_samples_per_video,
            taken_frames=center_frames,
            window_size=window_size // 2,
            frames_num=frame_count,
        )
        all_frames = np.concatenate([center_frames, negative_samples_frames])
        video_ids += [video_id] * len(all_frames)
        events += df["event"].to_list() + ["nothing"] * len(negative_samples_frames)
        fps_values += [fps] * len(all_frames)
        frame_counts += [frame_count] * len(all_frames)
        for i, frame_number in enumerate(all_frames):
            frames, clip_frame = extract_frames(
                cap,
                frame_number=int(frame_number),
                num_frames_to_extract=window_size,
            )
            if frame_size is not None:
                frames = [cv2.resize(frame, frame_size) for frame in frames]
            id = f"{video_id}_{i}"
            save_frames(
                frames,
                fps=fps,
                filename=output_dir / f"{id}.mp4",
            )
            clip_ids.append(id)
            times.append(frame_number / fps)
            clip_frames.append(clip_frame)
            t.update()

    new_df = pd.DataFrame(
        data={
            "video_id": video_ids,
            "clip_id": clip_ids,
            "time": times,
            "fps": fps_values,
            "frame_count": frame_counts,
            "clip_frame": clip_frames,
            "event": events,
        }
    )
    new_df.to_csv(output_dir / "labels.csv", index=False)

    t.close()


@app.command(
    help=(
        "Splits the created dataset into train and test sets. "
        "Must be used after the `extract` command."
    )
)
def split(
    test_size: float = typer.Option(default=0.2),
    random_state: Optional[int] = typer.Option(default=None),
    data_dir: Path = typer.Option(exists=True, default=Path("data", "processed")),
):
    """Splits the labels.csv file in the given directory into train and test sets.
    It assumes that it was previously generated using the `extract` command.
    """
    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        print(f"{labels_path} file does not exist. Run the `extract` command first.")
        typer.Exit(code=1)

    labels = pd.read_csv(labels_path)
    video_ids = labels["video_id"].unique()

    if test_size < 1:
        test_size = int(test_size * len(video_ids))

    indices = np.array(range(len(video_ids)))
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(indices)
    test_indices = indices[: int(test_size)]
    train_indices = indices[int(test_size) :]

    test_videos, train_videos = video_ids[test_indices], video_ids[train_indices]
    test_df = labels[labels["video_id"].isin(test_videos)]
    train_df = labels[labels["video_id"].isin(train_videos)]

    test_df.to_csv(data_dir / "test_labels.csv", index=False)
    train_df.to_csv(data_dir / "train_labels.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    app()
