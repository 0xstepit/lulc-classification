import dataclasses

import pytest
import torch
from torch import nn

from lulc.models.unet import DoubleConv, UNet, UNetConfig

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
            # Pyright flags this statically, which is the point of the test.
            UNetConfig().num_classes = 3  # pyright: ignore[reportGeneralTypeIssues]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"in_channels": 0},
            {"num_classes": 1},
            {"encoder_channels": ()},
            {"encoder_channels": (64, 0)},
        ],
    )
    def test_invalid_values_are_rejected(self, overrides):
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
            # Iterating an nn.ModuleList yields Module, which loses the
            # DoubleConv type. The assert narrows it and checks the wiring.
            assert isinstance(block, DoubleConv)

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


class TestReproducibility:
    def test_state_dict_round_trips_including_batchnorm_buffers(self, small_model):
        # Run a few batches in train mode so the BatchNorm running statistics
        # move away from their (0, 1) initialisation; otherwise a checkpoint
        # that dropped them would still reproduce the same predictions.
        small_model.train()
        for _ in range(3):
            small_model(torch.randn(2, 3, 32, 32))

        restored = UNet(SMALL_CONFIG)
        restored.load_state_dict(small_model.state_dict())

        small_model.eval()
        restored.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), small_model(x))

    def test_batchnorm_buffers_are_in_the_state_dict(self, small_model):
        keys = small_model.state_dict().keys()

        # named_parameters() does not see these, but --resume depends on them.
        assert any(k.endswith("running_mean") for k in keys)
        assert any(k.endswith("running_var") for k in keys)

    def test_running_statistics_only_update_in_train_mode(self, small_model):
        small_model.eval()
        batch_norm = small_model.bottleneck.block[1]
        before = batch_norm.running_mean.clone()

        with torch.no_grad():
            small_model(torch.randn(2, 3, 32, 32))

        # Catches a validation loop that forgot model.eval(), which corrupts the
        # running statistics with validation data and leaks it into inference.
        torch.testing.assert_close(batch_norm.running_mean, before)


class TestGradients:
    def test_backward_reaches_every_parameter(self, small_model):
        logits = small_model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 4, (2, 32, 32))

        nn.functional.cross_entropy(logits, target).backward()

        unreached = [
            name for name, p in small_model.named_parameters() if p.grad is None
        ]
        assert unreached == []

    def test_gradients_are_finite(self, small_model):
        logits = small_model(torch.randn(2, 3, 32, 32))
        target = torch.randint(0, 4, (2, 32, 32))

        nn.functional.cross_entropy(logits, target).backward()

        for name, p in small_model.named_parameters():
            assert torch.isfinite(p.grad).all(), name

    def test_ignore_index_pixels_do_not_block_learning(self, small_model):
        # 255 is the nodata label. A fully ignored batch has no signal, but a
        # partially ignored one must still produce gradients.
        logits = small_model(torch.randn(2, 3, 32, 32))
        target = torch.full((2, 32, 32), 255)
        target[:, :16, :] = torch.randint(0, 4, (2, 16, 32))

        loss = nn.functional.cross_entropy(logits, target, ignore_index=255)
        loss.backward()

        assert torch.isfinite(loss)
        assert small_model.head.weight.grad.abs().sum() > 0
