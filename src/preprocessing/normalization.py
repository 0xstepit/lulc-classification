import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np


@dataclass(frozen=True)
class NormalizationParams:
    low: np.ndarray
    high: np.ndarray
    median: np.ndarray

    def __post_init__(self) -> None:
        if not np.all(self.low.shape == self.high.shape == self.median.shape):
            raise ValueError("low, high, and median must have the same shape")

        # We should have one value per channel, so only 1D.
        if self.low.ndim != 1:
            raise ValueError("normalization stats must be one dimensional")

        if np.any(self.low > self.high):
            raise ValueError(
                "obtained low percentile values higher than high percentiles"
            )

    @classmethod
    def from_json(cls, path: Path) -> Self:
        with path.open() as fp:
            params = json.load(fp)

        return cls(
            low=np.asarray(params["low"], dtype=np.float32),
            high=np.asarray(params["high"], dtype=np.float32),
            median=np.asarray(params["median"], dtype=np.float32),
        )

    @property
    def num_channels(self) -> int:
        return self.low.shape[0]

    @property
    def percentile_range(self) -> np.ndarray:
        return self.high - self.low

    @property
    def normalized_median(self) -> np.ndarray:
        # Clip medians outliers to percentile boundaries.
        clipped_median = np.clip(self.median, self.low, self.high)
        return _scale(clipped_median, self)


# Expect data clipped to min and max
def _scale(data: np.ndarray, params: NormalizationParams) -> np.ndarray:
    perc_range = params.percentile_range
    denom = np.divide(
        1.0, perc_range, out=np.zeros_like(perc_range), where=perc_range > 0
    )
    return ((data - params.low) * denom).astype(np.float32)


def apply_normalization(
    patch: np.ndarray,
    params: NormalizationParams,
) -> np.ndarray:
    if patch.ndim != 3:
        raise ValueError(
            f"patch must have 3 dimensions (C, H, W), obtained ({patch.ndim})"
        )

    if patch.shape[0] != params.num_channels:
        raise ValueError(
            f"patch channel dimension must have same statistics dimension, "
            f"expected ({params.num_channels}) but got ({patch.shape[0]})"
        )

    # This way numpy can automatically broadcast the arrays.
    low = params.low[:, np.newaxis, np.newaxis]
    high = params.high[:, np.newaxis, np.newaxis]
    median = params.normalized_median[:, np.newaxis, np.newaxis]

    perc_range = high - low
    denom = np.divide(
        1.0, perc_range, out=np.zeros_like(perc_range), where=perc_range > 0
    )

    clipped_patch = np.clip(patch.astype(np.float32), low, high)
    normalized_patch = (clipped_patch - low) * denom

    # It could be better to fill NaN with neighboors?
    return np.where(np.isnan(normalized_patch), median, normalized_patch).astype(
        np.float32
    )
