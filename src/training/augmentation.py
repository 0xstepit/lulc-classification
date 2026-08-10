from dataclasses import dataclass
from multiprocessing import AuthenticationError

import torch


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_90: bool = True
    brightness_jitter: float = 0.05
    contrast_jitter: float = 0.05

    def __post_init__(self):
        if not 0.0 <= self.brightness_jitter <= 1:
            raise ValueError(
                f"brightness_jitter must be in [0, 1], obtained {self.brightness_jitter}"
            )

        if not 0.0 <= self.contrast_jitter <= 1:
            raise ValueError(
                f"contrast_jitter must be in [0, 1], obtained {self.brightness_jitter}"
            )


SAMPLING_PROBABILITY = 0.5


class PatchAugmentation:
    def __init__(
        self, cfg: AugmentationConfig, generator: torch.Generator | None = None
    ) -> None:
        self.cfg = cfg
        self.generator = generator

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self.generator).item())

    def _uniform(self, half_width: float, size: int) -> torch.Tensor:
        offset = torch.rand(size, generator=self.generator) * 2.0 - 1.0
        return 1.0 + offset * half_width

    def __call__(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment the ... with a pair of feature and labels tensors of size ([C, H, W], [H, W])"""
        # Dihedral group of the images:
        # Flip on the last dimension (W)
        if self.cfg.horizontal_flip and self._rand() < SAMPLING_PROBABILITY:
            features = torch.flip(features, dims=[-1])
            labels = torch.flip(labels, dims=[-1])

        # Flip on the second to last dimension (H)
        if self.cfg.vertical_flip and self._rand() < SAMPLING_PROBABILITY:
            features = torch.flip(features, dims=[-2])
            labels = torch.flip(labels, dims=[-2])

        if self.cfg.rotate_90:
            turns = int(torch.randint(0, 4, (1,), generator=self.generator).item())
            if turns:
                features = torch.rot90(features, turns, dims=(-2, -1))
                labels = torch.rot90(labels, turns, dims=(-2, -1))

        return features.clamp(0.0, 1.0).contiguous(), labels.contiguous()
