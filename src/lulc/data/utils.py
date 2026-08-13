import numpy as np


def evenly_spaced_indexes(max_index: int, wanted_indexes: int) -> list[int]:
    """Evenly select `wanted_indexes` indexes out of a total of possible `max_index`.
    This function can be used to evenly select a fixed number of STAC scene ordered by date.

    Parameters
    ----------
    max_index : int
        Defines the max index starting from 0 with step 1.
    wanted_indexes : int
        The number of desired indexes.

    Returns
    -------
    list[int]
        The list of indexes evenly spaced from 0 to `max_index`.
    """
    if wanted_indexes == 1:
        return [0]

    if wanted_indexes >= max_index:
        return list(range(max_index))

    step = (max_index - 1) / (wanted_indexes - 1)

    return [round(i * step) for i in range(wanted_indexes)]


def compute_nan_pct(scene: np.ndarray) -> np.float16:
    return round(
        np.count_nonzero(np.isnan(scene)) / scene.size * 100,
        2,
    ).astype(np.float16)
