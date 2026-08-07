from pathlib import Path

from src.constants import SEASONS_ORDER

ROOT = Path(__file__).parent.parent

# Configuration directories.
CONFIG_DIR = ROOT / "config"

REPORTER_CONFIG = CONFIG_DIR / "reporter.toml"
GLOBAL_CONFIG = CONFIG_DIR / "analysis.toml"

# Folder containing all the data used in the project.
DATA_DIR = ROOT / "data"

# Scenes directories

# Folder containing raw scene generated or downloaded from Sentinel2 data.
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Folder containing automatic reports results.
REPORTS_DIR = DATA_DIR / "reports"

# Suffixes are used when the name of final scene is composed
# with other strings.
SEASONAL_SCENE_SUFFIX = "_SEASONAL.tif"
ALL_BANDS_SCENE_SUFFIX = "_ALLBANDS.tif"

# Directories and files for the data ground truth labels.
LABELS_DIR = DATA_DIR / "labels"
WORLDCOVER_RAW_DIR = LABELS_DIR / "worldcover_raw"
WORLDCOVER_LABELS = LABELS_DIR / "worldcover_classes.tif"

PATCHES_DIR = DATA_DIR / "patches"
NORMALIZATION_PARAMS = PATCHES_DIR / "normalization_params.json"
PATCHES_METADATA = PATCHES_DIR / "patches_metadata.json"

# Seasonal scenes folder.
SEASONAL_SCENE_DIR = PROCESSED_DATA_DIR / "seasonal"
SEASONAL_SCENES = [
    SEASONAL_SCENE_DIR / f"{season}{SEASONAL_SCENE_SUFFIX}" for season in SEASONS_ORDER
]

IMGS_DIR = ROOT / "assets" / "images"
