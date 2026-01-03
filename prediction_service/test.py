import requests


payload  = {
    "id": 69,
    "host_id": 219517861,
  "host_response_rate": 1.0,
  "host_acceptance_rate": 1.0,
  "host_is_superhost": 0,
  "host_listings_count": 109.0,
  "host_total_listings_count": 119.0,
  "host_verifications": 3.0,
  "latitude": 37.97536,
  "longitude": 23.73172,
  "accommodates": 2.0,
  "bathrooms": 1.0,
  "bedrooms": 1.0,
  "beds": 2.0,
  "price": 72.0,
  "number_of_reviews": 22.0,
  "instant_bookable": 1,
  "calculated_host_listings_count": 70.0,
  "reviews_per_month": 0.53,
  "total_bookings": 4.0,
  "target": 1.0
}

url = "http://localhost:5000/predict/binary"


response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
