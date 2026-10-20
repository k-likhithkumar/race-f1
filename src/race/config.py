"""
Configuration and filesystem path settings for the RACE package.
"""

import os
from pathlib import Path

from typing import Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


def load_dotenv_file(env_path: Optional[Path] = None) -> None:
    """
    Loads environment variables from a .env file into os.environ.
    Does not overwrite existing environment variables.
    """
    if env_path is None:
        env_path = BASE_DIR / ".env"
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


# Ensure .env is read at startup if present
load_dotenv_file()


def get_gemini_api_key() -> str:
    """
    Returns the active Gemini API key, attempting to load from .env if not found in os.environ.
    Never logs or exposes the key value.
    """
    if "GEMINI_API_KEY" not in os.environ or not os.environ["GEMINI_API_KEY"]:
        load_dotenv_file()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    # If the key is an NVIDIA key but no NVIDIA_API_KEY is configured, do not treat as Gemini key
    if key.startswith("nvapi-"):
        return ""
    return key


def get_gemini_model_name() -> str:
    """Returns the configured Gemini model name, defaulting to active gemini-3.5-flash."""
    raw = os.environ.get("GEMINI_MODEL", os.environ.get("GEMINI_MODEL_NAME", "gemini-3.5-flash")).strip()
    if raw == "gemini-2.5-flash":
        return "gemini-3.5-flash"
    return raw


def get_nvidia_api_key() -> str:
    """
    Returns the active NVIDIA API key, checking NVIDIA_API_KEY or fallback to GEMINI_API_KEY if nvapi prefix.
    Never logs or exposes the key value.
    """
    if "NVIDIA_API_KEY" not in os.environ or not os.environ["NVIDIA_API_KEY"]:
        load_dotenv_file()
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        # Fallback if user placed nvapi key into GEMINI_API_KEY
        candidate = os.environ.get("GEMINI_API_KEY", "").strip()
        if candidate.startswith("nvapi-"):
            key = candidate
    return key


def get_nvidia_model_name() -> str:
    """Returns the configured NVIDIA model name, defaulting to verified meta/llama-3.2-11b-vision-instruct."""
    return os.environ.get("NVIDIA_MODEL", os.environ.get("NVIDIA_MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")).strip()


def get_nvidia_base_url() -> str:
    """Returns the configured NVIDIA API base URL, defaulting to https://integrate.api.nvidia.com/v1."""
    return os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip().rstrip("/")


def get_ai_provider_name() -> str:
    """
    Returns the active AI provider: 'gemini', 'nvidia', or 'mock'.
    Explicit AI_PROVIDER takes precedence over automatic detection.
    Automatic detection checks key prefixes (nvapi-... -> nvidia, AIza... -> gemini).
    """
    explicit = os.environ.get("AI_PROVIDER", "").strip().lower()
    if explicit:
        return explicit

    # Auto-detection
    nv_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if nv_key:
        return "nvidia"

    gem_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gem_key.startswith("nvapi-"):
        return "nvidia"
    if gem_key.startswith("AIza"):
        return "gemini"

    if gem_key:
        return "gemini"

    return "gemini"


# AI Strategy Assistant Configuration
AI_PROVIDER = get_ai_provider_name()
GEMINI_API_KEY = get_gemini_api_key()
GEMINI_MODEL_NAME = get_gemini_model_name()
NVIDIA_API_KEY = get_nvidia_api_key()
NVIDIA_MODEL_NAME = get_nvidia_model_name()
NVIDIA_BASE_URL = get_nvidia_base_url()

AI_REQUEST_TIMEOUT_S = float(os.environ.get("AI_REQUEST_TIMEOUT_S", "35.0"))
AI_MAX_TOOL_TURNS = 3
AI_MAX_PROMPT_CHARS = 1000
AI_MAX_HISTORY_MESSAGES = 10

SUPPORTED_GEMINI_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
}

SUPPORTED_NVIDIA_MODELS = {
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-large-2-instruct",
}

# Separation of raw network cache and normalized Parquet cache
RAW_FASTF1_CACHE_DIR = DATA_DIR / "raw_fastf1"
RAW_OPENF1_CACHE_DIR = DATA_DIR / "raw_openf1"
PARQUET_CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"

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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_race_slug(year: int, gp: str) -> str:
    """Generate a clean filesystem slug for a Grand Prix session."""
    clean_gp = gp.lower().replace(" ", "_").replace("-", "_")
    return f"{year}_{clean_gp}"


def get_race_cache_dir(year: int, gp: str) -> Path:
    """Return the Parquet cache directory path for a specific race."""
    return PARQUET_CACHE_DIR / get_race_slug(year, gp)
