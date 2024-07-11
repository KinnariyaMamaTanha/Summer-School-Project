from torch import nn
from torch.nn import init
import torch


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        init.kaiming_normal_(self.conv1.weight, a=1e-4)
        self.conv1.bias.data.zero_()  # type: ignore
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=1e-4, inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        init.kaiming_normal_(self.conv2.weight)
        self.conv2.bias.data.zero_()  # type: ignore
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        id = x

        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)

        o = self.conv2(o)
        o = self.bn2(o)

        if self.downsample is not None:
            id = self.downsample(x)

        o += id
        o = self.relu(o)

        return o


class AudioClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = self._make_layer(
            in_channels=2, out_channels=8, kernel_size=5, stride=2
        )
        self.layer2 = self._make_layer(
            in_channels=8, out_channels=16, kernel_size=5, stride=2
        )
        self.layer3 = self._make_layer(
            in_channels=16, out_channels=32, kernel_size=3, stride=2
        )
        self.layer4 = self._make_layer(
            in_channels=32, out_channels=64, kernel_size=3, stride=2
        )
        self.layer5 = self._make_layer(
            in_channels=64, out_channels=64, kernel_size=3, stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, in_channels, out_channels, kernel_size=3, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            tmp = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            init.kaiming_normal_(tmp.weight, a=1e-4)

            downsample = nn.Sequential(
                tmp,
                nn.BatchNorm2d(out_channels),
            )
        return ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            downsample=downsample,
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
