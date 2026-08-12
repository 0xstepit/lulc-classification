import dataclasses

import pytest
import torch
from torch import nn

from src.models.unet import DoubleConv, UNet, UNetConfig

EXPECTED_PARAMS = 31_066_312

SMALL_CONFIG = UNetConfig(in_channels=3, num_classes=4, encoder_channels=(8, 16))


@pytest.fixture
def small_model() -> UNet:
    torch.manual_seed(3)
    return UNet(SMALL_CONFIG)


class TestUNetConfig:
    def test_defaults(self):
        cfg = UNetConfig()

        assert cfg.in_channels == 52
        assert cfg.num_classes == 8
        assert cfg.depth == 4
        assert cfg.bottleneck_channels == 1024
        assert cfg.size_divisor == 16

    def test_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            UNetConfig().num_classes = 3

    @pytest.mark.parametrize(
        "overrides",
        [
            {"in_channels": 0},
            {"num_classes": 1},
            {"encoder_channels": ()},
            {"encoder_channels": (64, 0)},
        ],
    )
    def test_invalid_values_are_rejectedc(self, overrides):
        with pytest.raises(ValueError):
            dataclasses.replace(UNetConfig(), **overrides)
