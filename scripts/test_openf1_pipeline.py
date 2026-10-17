"""
Test prototype for OpenF1 pipeline on 2023 Monza.
"""

import os
import json
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_OPENF1_DIR = Path("data/raw_openf1")
RAW_OPENF1_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://api.openf1.org/v1"

def fetch_json(endpoint: str, cache_name: str, params: dict = None):
    cache_file = RAW_OPENF1_DIR / f"{cache_name}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data

if __name__ == "__main__":
    session_key = 9157
    print("Testing OpenF1 data extraction for Monza 2023...")
    sess = fetch_json("sessions", "sess_9157", {"session_key": session_key})[0]
    print("Session:", sess["circuit_short_name"], sess["year"], sess["date_start"])
    
    drivers = fetch_json("drivers", "drivers_9157", {"session_key": session_key})
    print("Drivers:", len(drivers))
    
    laps = fetch_json("laps", "laps_9157", {"session_key": session_key})
    print("Total laps:", len(laps))
    
    stints = fetch_json("stints", "stints_9157", {"session_key": session_key})
    print("Total stints:", len(stints))
    
    pits = fetch_json("pit", "pit_9157", {"session_key": session_key})
    print("Total pits:", len(pits))
    
    weather = fetch_json("weather", "weather_9157", {"session_key": session_key})
    print("Total weather records:", len(weather))
    
    rc = fetch_json("race_control", "rc_9157", {"session_key": session_key})
    print("Total race control events:", len(rc))
