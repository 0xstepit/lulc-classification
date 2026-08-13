from dataclasses import dataclass
from math import cos, radians
from typing import TypeAlias

from rasterio import Affine, warp
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform

# Helper alias for  w | s | e | n
BoundingBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True)
class GCS:
    """Geographic Coordinate System class.

    Attributes
    ----------
    latitude : float
        The values of the latitude in degrees.
    longitude : float
        The values of the longitude in degrees.
    """

    latitude: float
    longitude: float


def deg_per_m(meters: float, center_lat: float) -> GCS:
    """Convert with approximation latitude and longitude from meters into degrees.
    Ref: https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance

    Parameters
    ----------
    meters : float
        The meters to convert.
    center_lat : float
        The latitude to use for the correction of the longitudinal degrees.

    Returns
    -------
    GCS

    """
    lat_deg = meters / 110_574
    lon_deg = meters / (111_320 * cos(radians(center_lat)))

    return GCS(lat_deg, lon_deg)


def create_bbox(gcs: GCS, size_m: float) -> BoundingBox:
    """Create a bounding box around the provided center of the specified size.
    Due to distortions in the projections of the Earth model, the resulting bounding box size
    will be close to but not equal to the specified size.

    Parameters
    ----------
    gcs : GCS
        Geograhic coordinates of the center of the box.
    size_m : float
        Size of each bounding box side in meters.

    Returns
    -------
    BoundingBox
        The bounding box around the provided center.

    """
    half_size = size_m / 2

    side_deg = deg_per_m(half_size, gcs.latitude)

    return (
        gcs.longitude - side_deg.longitude,
        gcs.latitude - side_deg.latitude,
        gcs.longitude + side_deg.longitude,
        gcs.latitude + side_deg.latitude,
    )


def create_window_and_transform_from_bbox(
    bbox: BoundingBox, crs, transform
) -> tuple[Window, Affine]:
    """Create a rasterio window from a bounding box and geometric information of the Items
    that will be windowed. The computed window values are rounded.

    Parameters
    ----------
    bbox : BoundingBox
        Bounding box defining the window defined in the EPSG:4326.
    crs :
        Rasterio Profile CRS.
    transform :
        Rasterio Profile transformation matrix.

    Returns
    -------
    tuple[Window, Affine]
        The Rasterio Window to crops a raster and its associated affine transformation matrix.
    """
    # Transform from degree to meters.
    left, bottom, right, top = warp.transform_bounds("EPSG:4326", crs, *bbox)

    window = from_bounds(
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        transform=transform,
    )
    # Get the smalles pixel-aligned window that fully contains the bounding box.
    window = window.round_offsets().round_lengths()

    cropped_transform = window_transform(window, transform)

    return window, cropped_transform
