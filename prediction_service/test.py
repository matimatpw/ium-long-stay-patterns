import requests

import json
from pathlib import Path

p = Path("prediction_service/test_data/test_targets_1.json")
with p.open("r", encoding="utf-8") as f:
    data = json.load(f)

# print(len(data))

url = "http://localhost:5000/predict/binary"
for record in data:

    response = requests.post(url, json=record)
    print(response.json(), end=f", {response.status_code}\n")

