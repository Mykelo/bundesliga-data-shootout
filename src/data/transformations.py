from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np


class VideoToTensor(object):
    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        # It expects frames at the first dimension
        tensor_frames: list[torch.Tensor] = []
        to_tensor = transforms.ToTensor()
        for frame in sample:
            tensor_frames.append(to_tensor(frame))

        return torch.stack(tensor_frames)


class VideoRandomHorizontalFlip(object):
    p: float

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # It expects frames at the first dimension
        tensor_frames: list[torch.Tensor] = []
        should_flip = random.random() < self.p
        for frame in sample:
            if should_flip:
                tensor_frames.append(TF.hflip(frame))
            else:
                tensor_frames.append(frame)

        return torch.stack(tensor_frames)


class VideoResize(object):
    size: tuple[int, int] | int
    interpolation: transforms.InterpolationMode
    max_size: int | None
    antialias: bool | None

    def __init__(
        self,
        size: tuple[int, int] | int,
        interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
        max_size: int | None = None,
        antialias: bool | None = None,
    ):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # It expects frames at the first dimension
        tensor_frames: list[torch.Tensor] = []
        resize = transforms.Resize(
            size=self.size,
            interpolation=self.interpolation,
            max_size=self.max_size,
            antialias=self.antialias,
        )
        for frame in sample:
            tensor_frames.append(resize(frame))

        return torch.stack(tensor_frames)
