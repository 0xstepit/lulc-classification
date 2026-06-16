from pathlib import Path

ROOT = Path(__file__).parent.parent

# Folder containing all the data used in the project.
DATA_DIR = ROOT / "data"
# Folder containing raw scene generated or downloaded from Sentinel2 data.
RAW_DIR = DATA_DIR / "raw"
# Folder containing automatic analysis results.
ANALYSIS_DIR = DATA_DIR / "analysis"

# Folder containing all the configuration files for the scripts.
CONFIG_DIR = ROOT / "config"
REPORTER_CONFIG = CONFIG_DIR / "reporter.toml"

# Name of the configuration file.
CONFIG_FILE = "config.toml"
