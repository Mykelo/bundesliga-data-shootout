import torch
from torch import nn
from torchvision import models


class R3DDFL(nn.Module):
    def __init__(self):
        super().__init__()
        self.r3d = models.video.r3d_18(
            weights=models.video.R3D_18_Weights.KINETICS400_V1
        )
        self.r3d.fc = nn.Linear(in_features=512, out_features=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.r3d(x.permute(0, 2, 1, 3, 4))
