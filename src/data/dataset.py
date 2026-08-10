import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.io import FEATURE_SUFFIX, LABEL_SUFFIX
from src.preprocessing.normalization import NormalizationParams, apply_normalization

# [[features, labels], tuple[transformed_features, transformed_labels]]
Transform = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]

logger = logging.getLogger(__name__)


class LULCPatchDataset(Dataset):
    def __init__(
        self,
        patches_dir: Path,
        params: NormalizationParams,
        transform: Transform | None = None,
    ) -> None:
        self.patches_dir = patches_dir
        self.params = params
        self.transform = transform

        self.feature_paths = sorted(patches_dir.glob(f"*{FEATURE_SUFFIX}"))
        if not self.feature_paths:
            raise FileNotFoundError(f"no patches found in {patches_dir}")

        self.label_paths = [
            p.with_name(p.name.replace(FEATURE_SUFFIX, LABEL_SUFFIX))
            for p in self.feature_paths
        ]

        # We need to check if all the label paths actually exist.
        missing = [p for p in self.label_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"feature patches are missing label patches ({missing})"
            )

        logger.info(f"loaded {len(self.feature_paths)} from {patches_dir}")

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = np.load(self.feature_paths[index])
        labels = np.load(self.label_paths[index])

        features = apply_normalization(features, self.params)

        feature_tensor = torch.from_numpy(features)
        #  CrossEntropyLoss requires int64
        label_tensor = torch.from_numpy(labels.astype(np.int64))

        if self.transform:
            feature_tensor, label_tensor = self.transform(feature_tensor, label_tensor)

        return feature_tensor, label_tensor
