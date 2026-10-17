"""
Diagnostic script to test OpenF1 independently for 2023 Monza (session_key 9157).
Tests all 9 required endpoints without authentication.
"""

import json
import requests

SESSION_KEY = 9157  # 2023 Italian Grand Prix - Race
BASE_URL = "https://api.openf1.org/v1"

endpoints = {
    "sessions": f"{BASE_URL}/sessions?session_key={SESSION_KEY}",
    "drivers": f"{BASE_URL}/drivers?session_key={SESSION_KEY}",
    "laps": f"{BASE_URL}/laps?session_key={SESSION_KEY}&driver_number=1",
    "car_data": f"{BASE_URL}/car_data?session_key={SESSION_KEY}&driver_number=1&speed>=300",
    "location": f"{BASE_URL}/location?session_key={SESSION_KEY}&driver_number=1&date>=2023-09-03T13:25:00&date<=2023-09-03T13:25:10",
    "pit": f"{BASE_URL}/pit?session_key={SESSION_KEY}",
    "stints": f"{BASE_URL}/stints?session_key={SESSION_KEY}",
    "weather": f"{BASE_URL}/weather?session_key={SESSION_KEY}",
    "race_control": f"{BASE_URL}/race_control?session_key={SESSION_KEY}",
    "position": f"{BASE_URL}/position?session_key={SESSION_KEY}&driver_number=1",
}

print(f"=== TESTING OPENF1 FOR SESSION {SESSION_KEY} (2023 Monza) ===")

results = {}

for name, url in endpoints.items():
    try:
        r = requests.get(url, timeout=15)
        status = r.status_code
        if status == 200:
            data = r.json()
            count = len(data)
            sample = data[0] if count > 0 else {}
            fields = list(sample.keys()) if sample else []
            results[name] = {
                "status": status,
                "count": count,
                "fields": fields,
                "sample": sample,
            }
            print(f"[OK] {name:12} -> HTTP 200, {count} records, fields: {fields[:6]}...")
        else:
            results[name] = {"status": status, "error": r.text}
            print(f"[FAIL] {name:12} -> HTTP {status}: {r.text[:100]}")
    except Exception as exc:
        results[name] = {"status": "ERROR", "error": str(exc)}
        print(f"[ERROR] {name:12} -> {exc}")

with open("data/openf1_diag_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nDiagnostic complete. Results written to data/openf1_diag_results.json")
