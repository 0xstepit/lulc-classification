"""U-Net model for semantic segmentation using Sentinel-2 seasonal composites."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 52
    num_classes: int = 8
    encoder_channels: tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(
                f"in_channels must be a positive number, got {self.in_channels}"
            )

        if self.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {self.num_classes}")

        if not self.encoder_channels:
            raise ValueError("encoder_channels must be provided, received empty")

        if any([chan < 1 for chan in self.encoder_channels]):
            raise ValueError(
                f"encode_channels must be all positive, got {self.encoder_channels}"
            )

    @property
    def depth(self) -> int:
        return len(self.encoder_channels)

    @property
    def bottleneck_channels(self) -> int:
        return self.encoder_channels[-1] * 2

    @property
    def size_divisor(self) -> int:
        return 2**self.depth


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UNetConfig()

        self.encoder_blocks = nn.ModuleList()
        width = self.cfg.in_channels
        for stage_width in self.cfg.encoder_channels:
            self.encoder_blocks.append(DoubleConv(width, stage_width))
            width = stage_width

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(width, self.cfg.bottleneck_channels)

        self.upsamplers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        width = self.cfg.bottleneck_channels
        for stage_width in reversed(self.cfg.encoder_channels):
            self.upsamplers.append(
                nn.ConvTranspose2d(width, stage_width, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(stage_width * 2, stage_width))
            width = stage_width

        self.head = nn.Conv2d(width, self.cfg.num_classes, kernel_size=1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected a [N, C, H, W] tensor, got {x.ndim} dimensions")

        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"expected {self.cfg.in_channels} input channels, got {x.shape[1]}"
            )

        divisor = self.cfg.size_divisor
        height, width = x.shape[-2:]
        if height % divisor or width % divisor:
            raise ValueError(
                f"spatial dimensions must be a multiple of {divisor} "
                f"got a {height}x{width} patch"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)

        skips = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, block, skip in zip(
            self.upsamplers, self.decoder_blocks, reversed(skips), strict=True
        ):
            x = upsample(x)
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.head(x)
