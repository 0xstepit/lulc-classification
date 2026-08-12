from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotate_90: bool = True


SAMPLING_PROBABILITY = 0.5


class PatchAugmentation:
    def __init__(
        self, cfg: AugmentationConfig, generator: torch.Generator | None = None
    ) -> None:
        self.cfg = cfg
        self.generator = generator

    def _rand(self) -> float:
        return float(torch.rand(1, generator=self.generator).item())

    def __call__(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Augment a pair of feature and label tensors of size ([C, H, W], [H, W])."""
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

        # No clamping: flips and rotations permute pixels without changing their
        # values, and apply_normalization already guarantees the [0, 1] range.
        return features.contiguous(), labels.contiguous()
