"""Simple A/B tester for naive (baseline) and binary model endpoints.

For each record in `test_data/test_ab_data.csv` the script randomly
chooses one of the two endpoints (/predict/base or /predict/binary), sends
POST, and appends a JSON line to `logs_ab.txt` with timestamp, chosen model,
record id (if present), response status and prediction (or error).

Usage: poetry run python prediction_service/ab_test.py
"""

import json
import random
import argparse
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd

from ium_long_stay_patterns.config import set_seed

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "test_data" / "test_ab_data.csv"
LOG_PATH = ROOT / "logs" / "logs_ab.txt"

URL_BASE = "http://localhost:5000/predict"
ENDPOINTS = {
    "naive": "/base",
    "binary": "/binary",
}

def load_data_csv(path: Path):
    df = pd.read_csv(path)
    return df.to_dict(orient='records')

def load_data(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def log_entry(entry: dict):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="A/B test runner")
    parser.add_argument("--seed", type=int, default=None, help="Global seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    data = load_data_csv(DATA_PATH)

    counts = {"naive": 0, "binary": 0, "all": 0}

    for rec in data:
        chosen = random.choice(list(ENDPOINTS.keys()))
        counts[chosen] += 1
        counts["all"] += 1
        endpoint = URL_BASE + ENDPOINTS[chosen]

        ts = datetime.utcnow().isoformat() + "Z"
        payload = rec
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            try:
                body = resp.json()
            except Exception:
                body = {"error": "Invalid JSON response", "text": resp.text}

            entry = {
                "timestamp": ts,
                "model": chosen,
                "prediction": body.get("prediction"),
                "request_id": rec.get("id"),
                "status_code": resp.status_code,
            }
        except Exception as e:
            entry = {
                "timestamp": ts,
                "model_chosen": chosen,
                "status_code": None,
                "request_id": rec.get("id"),
                "response": {"error": str(e)},
            }

        log_entry(entry)
        print(f"{ts} - {chosen} - id={rec.get('id')} - status={entry['status_code']}")

    print("\nSummary:")
    for k, v in counts.items():
        print(f"  {k}: {v} requests")


if __name__ == "__main__":
    main()
