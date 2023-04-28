import torch
from torch import nn
from torchvision import models


class R3DDFL(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.r3d = models.video.r3d_18(
            weights=models.video.R3D_18_Weights.KINETICS400_V1
        )
        self.r3d.fc = nn.Linear(in_features=512, out_features=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.r3d(x.permute(0, 2, 1, 3, 4))


class ResNetLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.5,
        lstm_hidden_size: int = 16,
        lstm_num_layers: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.resnet = models.resnet18(weights=models.resnet.ResNet18_Weights)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(num_features, lstm_hidden_size, lstm_num_layers)
        self.fc1 = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, c, h, w = x.shape

        features = []
        for i in range(frames):
            z_hat = self.resnet(x[:, i])
            features.append(z_hat)

        stacked = torch.stack(features, dim=1)
        out, (hn, cn) = self.lstm(stacked)

        out = self.dropout(out[:, -1])
        out = self.fc1(out)
        return out
