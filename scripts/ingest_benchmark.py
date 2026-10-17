"""
Script to ingest and cache the benchmark race (2023 Monza).
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.race.data.cache import cache_race, is_race_cached

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    year = 2022
    gp = "Monza"
    print(f"--- Ingesting Benchmark Race: {year} {gp} ---")
    summary = cache_race(year, gp)
    print("\n--- Cache Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nis_race_cached:", is_race_cached(year, gp))
