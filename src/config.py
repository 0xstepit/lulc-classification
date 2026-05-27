import tomllib
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"


def deep_merge(base: dict, overrides: dict) -> dict:
    """
    Right join two configuration dictionaries.

    Parameters
    ----------
    base: dict
        Base configuration dictionary.
    overrides: dict
        The customized configuration dictionary.

    Returns
    -------
    merged: dict
        The right join of the two provided configuration dictionaries.

    """
    merged = base.copy()
    for key, value in overrides.items():
        # If the ooverride value is a dict and the base is dict, call recurrently. If not,
        # just copy and override.
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(files: list[str]) -> dict:
    """
    Main configurations loader.

    Parameters
    ----------
    files: list[str]
        The names of the configuration files to load

    Returns
    -------
    config: dict
        The merged configuration containing all the fields of provided files.

    Raises
    ------
    FileNotFoundError
        If one of the file in the input does not exist.

    """
    config = {}

    for file in files:
        file_path = CONFIG_DIR / file
        if not file_path.exists():
            raise FileNotFoundError(
                f"File named {file} does not exists in {CONFIG_DIR}"
            )

        with open(file_path, "rb") as f:
            tmp_config = tomllib.load(f)

        config = deep_merge(config, tmp_config)

    return config
