from pathlib import Path
import numpy as np
import cv2


def extract_frames(
    cap: cv2.VideoCapture, frame_number: int, num_frames_to_extract: int
) -> tuple[list[np.ndarray], int]:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, frame_number - num_frames_to_extract // 2)
    end_frame = min(frame_count - 1, frame_number + num_frames_to_extract // 2)
    frames: list[np.ndarray] = []
    for i in range(int(start_frame), int(end_frame) + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames, frame_number - start_frame


def save_frames(frames: list[np.ndarray], fps: int, filename: Path):
    out = cv2.VideoWriter(
        str(filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        out.write(frame)
    out.release()


def gaussian(x: np.ndarray, mu: float, sig: float):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def create_label(
    frames: list[int],
    length: int,
    window_size: int = 11,
    eps: float = 1e-1,
) -> np.ndarray:
    window = np.array(range(window_size))
    window = gaussian(window, window_size // 2, 2.5)
    window[window < eps] = 0

    label = np.zeros(length)
    for frame in frames:
        label[frame - window_size // 2 : frame + window_size // 2 + 1] += window

    label[label > 1] = 1
    return label


def mark_frames(mask: np.ndarray, frame: int, window_size: int) -> np.ndarray:
    new_mask = mask.copy()
    start_frame = max(0, frame - window_size // 2)
    end_frame = frame + window_size // 2
    new_mask[start_frame : end_frame + 1] = False
    return new_mask


def sample_negative(
    sample_size: int, taken_frames: np.ndarray, window_size: int, frames_num: int
) -> np.ndarray:
    frames = np.array(list(range(frames_num)))
    available_frames_mask = np.array([True] * frames_num)

    # First mark all taken frames as taken
    for frame in taken_frames:
        available_frames_mask = mark_frames(
            available_frames_mask, frame=int(frame), window_size=window_size
        )

    available_frames = frames[available_frames_mask]
    # Filter out some more frames to prevent too similar clips
    available_frames = available_frames[:: window_size // 2]
    sampled_frames = np.random.choice(available_frames, sample_size, replace=False)

    return sampled_frames


def read_video_to_numpy(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []

    # loop through all frames in the video
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break

        frames.append(frame)

    # release the video capture object
    cap.release()

    # convert the list of frames to a NumPy array
    video = np.array(frames)

    return video
