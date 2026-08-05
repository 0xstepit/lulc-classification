import math
from dataclasses import dataclass

SET_NAMES = ["train", "val", "test"]


@dataclass(frozen=True)
class PatchesConfig:
    seed: int
    seed_candidates: int
    size: int
    block_size: int
    buffer: int
    max_nan_fraction: float
    normalization_percentiles: list[float]
    split: dict[str, float]

    def __post_init__(self) -> None:
        if self.size == 0:
            raise ValueError("patch size must be greater than zero")

        if self.block_size % self.size != 0:
            raise ValueError(
                f"block_size {self.block_size} must be an integer multiple of"
                f"patch size {self.size}"
            )

        if set(self.split) != set(SET_NAMES):
            raise ValueError(f"split names must be {SET_NAMES}")

        if not math.isclose(sum(self.split.values()), 1.0):
            raise ValueError("the total sum of set percentage division must be 1.0")

        if len(self.normalization_percentiles) != 2:
            raise ValueError("normalization_percentiles must be [low, high]")

        low, high = self.normalization_percentiles
        if not (0 <= low and high <= 100):
            raise ValueError(
                "normalization percentiles must be 0 <= low and high <= 100"
            )

        if not 0.0 <= self.max_nan_fraction <= 1.0:
            raise ValueError("max_nan_fraction must be in [0, 1]")

    @property
    def patches_per_block(self) -> int:
        return self.block_size // self.size
