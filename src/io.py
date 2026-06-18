from pathlib import Path

ROOT = Path(__file__).parent.parent

# Configuration directories.
CONFIG_DIR = ROOT / "config"

REPORTER_CONFIG = CONFIG_DIR / "reporter.toml"
GLOBAL_CONFIG = CONFIG_DIR / "config.toml"

# Folder containing all the data used in the project.
DATA_DIR = ROOT / "data"

# Scenes directories

# Folder containing raw scene generated or downloaded from Sentinel2 data.
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Seasonal scenes folder.
SEASONAL_SCENE_DIR = PROCESSED_DATA_DIR / "seasonal"
MULTISEASONAL_SCENE_DIR = PROCESSED_DATA_DIR / "multiseasonal"

# Analysis directories
# Folder containing automatic analysis results.
ANALYSIS_DIR = DATA_DIR / "analysis"

# Suffixes are used when the name of final scene is composed
# with other strings.
SEASONAL_SCENE_SUFFIX = "_SEASONAL.tif"
ALL_BANDS_SCENE_SUFFIX = "_ALLBANDS.tif"
