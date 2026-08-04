from dataclasses import dataclass

from numpy import size


@dataclass(frozen=True)
class PatchesConfig:
    seed: int
    size: int
    block_size: int
    buffer: int
    split: dict[str, float]

    def __post_init__(self) -> None:
        if self.size == 0:
            raise ValueError("patch size must be greater than zero")

        if self.block_size % self.size != 0:
            raise ValueError("block_size must be an integer multiple of patch size")

        cum_sum = 0
        set_num = 0
        for v in self.split.values():
            set_num += 1
            cum_sum += v

        if set_num != 2:
            raise ValueError("there should be three different sets")

        if cum_sum != 1.0:
            raise ValueError("the total sum of set percentage division must be 1.0")
