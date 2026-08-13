import numpy as np

IGNORE_IDX = np.iinfo(np.uint8).max


def build_remap_lookup_table(
    labels: np.ndarray, nodata_value: int = 0, table_size: int = 256
) -> tuple[np.ndarray, dict[int, int]]:
    """Create a lookup table that can be used to remap the classes in the labels raster
    into a consecutive sequence starting from 0.

    Parameters
    ----------
    labels : np.ndarray
        The non-consecutive labels raster.
    nodata_value : int
        The value in the labels raster associated with no data.
    table_size : int
        The size of the lookup table. This value shuold be equal to the highest value
        that has to be represented in the raster. As a default, it is used the max uint8 + 1.

    Returns
    -------
        lut : np.ndarray
            The lookup table that can be used to remap the mapping key into its value.
        mapping : dict[int, int]
            A mapping from the original value to its sequential index.
    """
    present_labels = [int(v) for v in np.unique(labels) if int(v) != nodata_value]

    mapping = {label: index for index, label in enumerate(present_labels)}

    lut = np.full(table_size, IGNORE_IDX, dtype=np.uint8)
    for value, index in mapping.items():
        lut[value] = index

    return lut, mapping
