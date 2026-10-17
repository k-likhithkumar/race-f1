"""
Configuration and filesystem path settings for the RACE package.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Separation of raw network cache and normalized Parquet cache
RAW_FASTF1_CACHE_DIR = DATA_DIR / "raw_fastf1"
RAW_OPENF1_CACHE_DIR = DATA_DIR / "raw_openf1"
PARQUET_CACHE_DIR = DATA_DIR / "cache"

# API Endpoints
OPENF1_BASE_URL = "https://api.openf1.org/v1"
JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Resampling frequency for interactive circuit map positions
POSITION_RESAMPLE_HZ = 5.0
POSITION_MAX_INTERPOLATION_GAP_S = 2.0

# Stage 2 Analytics Constants
FUEL_CORRECTION_S_PER_LAP = 0.033  # Standard linear fuel burn pace benefit (s/lap)
BATTLE_GAP_THRESHOLD_S = 1.0       # Proximity threshold for active battle detection (s)
BATTLE_CLOSING_RATE_THRESHOLD_S = 0.05  # Closing rate threshold for active battle (s/lap)
TRAFFIC_GAP_THRESHOLD_S = 1.2      # Wake dirty air threshold (s)
MINI_SECTOR_ACCURACY_TOLERANCE_S = 0.05  # Maximum discrepancy tolerance between sum and lap time (s)
DEFAULT_NUM_MINI_SECTORS = 20      # Standard mini-sectors per lap per PDF spec


def ensure_directories() -> None:
    """Ensure data cache directories exist."""
    RAW_FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OPENF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_race_slug(year: int, gp: str) -> str:
    """Generate a clean filesystem slug for a Grand Prix session."""
    clean_gp = gp.lower().replace(" ", "_").replace("-", "_")
    return f"{year}_{clean_gp}"


def get_race_cache_dir(year: int, gp: str) -> Path:
    """Return the Parquet cache directory path for a specific race."""
    return PARQUET_CACHE_DIR / get_race_slug(year, gp)
