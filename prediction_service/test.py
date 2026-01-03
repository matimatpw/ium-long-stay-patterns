import requests


payload  = {
  "id": 30419466,
  "host_id": 135482103,
  "host_response_rate": 0.86,
  "host_acceptance_rate": 1.0,
  "host_is_superhost": 1,
  "host_listings_count": 20,
  "host_total_listings_count": 24,
  "host_verifications": 2,
  "latitude": 37.97251,
  "longitude": 23.72772,
  "accommodates": 10,
  "bathrooms": 3.0,
  "bedrooms": 4.0,
  "beds": 7.0,
  "price": 432.0,
  "number_of_reviews": 181,
  "instant_bookable": 1,
  "calculated_host_listings_count": 12,
  "reviews_per_month": 2.51
}

url = "http://localhost:5000/predict"


response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
