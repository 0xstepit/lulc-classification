import dataclasses
from ast import mod

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


class TestForward:
    def test_full_model_maps_52_channels_to_8_classes(self):
        model = UNet()

        output = model(torch.rand(3, 52, 256, 256))

        assert tuple(output.shape) == (3, 8, 256, 256)

    def test_spatial_dimensions_are_preserved(self, small_model):

        output = small_model(torch.randn(3, 3, 32, 32))

        assert tuple(output.shape) == (3, 4, 32, 32)

    def test_non_square_input_is_accepted(self, small_model):
        output = small_model(torch.randn(1, 3, 32, 64))

        assert tuple(output.shape) == (1, 4, 32, 64)


class TestArchitecture:
    def test_parameters_count_is_stable(self):
        assert UNet().num_parameters == EXPECTED_PARAMS

    def test_decoder_blocks_consume_the_concatenated_skips(self):
        model = UNet()

        for block, stage_width in zip(
            model.decoder_blocks, reversed(UNetConfig().encoder_channels), strict=True
        ):
            first_conv = block.block[0]
            assert first_conv.in_channels == stage_width * 2
            assert first_conv.out_channels == stage_width

    def test_convolutions_before_batchnorm_carry_no_bias(self):
        model = UNet(SMALL_CONFIG)

        for module in model.modules():
            if isinstance(module, DoubleConv):
                assert module.block[0].bias is None
                assert module.block[3].bias is None

    def test_depth_follows_the_encoder_configuration(self):
        model = UNet(
            UNetConfig(in_channels=3, num_classes=2, encoder_channels=(4, 8, 16))
        )

        assert len(model.encoder_blocks) == 3
        assert len(model.decoder_blocks) == 3
        assert model.cfg.size_divisor == 8
        assert tuple(model(torch.randn(1, 3, 24, 24)).shape) == (1, 2, 24, 24)
