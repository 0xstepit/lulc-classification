def normalize_rgb(rgb: list[int]):
    """Normalize the RGB values into the range [0, 1] by dividing by 255.

    Parameters
    ----------
    rgb :
        A list containing the three RGB values

    Returns
    -------
        The RGB values rescaled into [0, 1]
    """
    return tuple([x / 255 for x in rgb])
