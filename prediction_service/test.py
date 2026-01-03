import requests

import json
from pathlib import Path

p = Path("prediction_service/test_data/test_targets_1.json")
with p.open("r", encoding="utf-8") as f:
    data = json.load(f)


# payload  = {
#     "id": 1241285021917918430,
#     "host_id": 303135507,
#     "host_response_rate": 1.0,
#     "host_acceptance_rate": 1.0,
#     "host_is_superhost": 0,
#     "host_listings_count": 2,
#     "host_total_listings_count": 2,
#     "host_verifications": 1,
#     "latitude": 37.9686824,
#     "longitude": 23.7499517,
#     "accommodates": 4,
#     "bathrooms": 1.0,
#     "bedrooms": 1.0,
#     "beds": 2.0,
#     "price": 44.0,
#     "number_of_reviews": 6,
#     "instant_bookable": 1,
#     "calculated_host_listings_count": 1,
#     "reviews_per_month": 2.34,
#     "listing_id": 1241285021917918430,
#     "total_bookings": 2
#   }



# print(len(data))

url = "http://localhost:5000/predict/binary"
for record in data:

    response = requests.post(url, json=record)
    print(response.json(), end=f", {response.status_code}\n")

