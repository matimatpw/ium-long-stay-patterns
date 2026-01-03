from flask import Flask, request, jsonify

app = Flask(__name__)

REQUIRED_FIELDS = [
    'id',
    'host_id',
    'host_response_rate',
    'host_acceptance_rate',
    'host_is_superhost',
    'host_listings_count',
    'host_total_listings_count',
    'host_verifications',
    'latitude',
    'longitude',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'price',
    'number_of_reviews',
    'instant_bookable',
    'calculated_host_listings_count',
    'reviews_per_month'
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    # Placeholder prediction
    prediction = 0
    probability = 0.5

    return jsonify({
        "id": data["id"],
        "prediction": prediction,
        "probability": probability
    }), 200


if __name__ == "__main__":
    # Use 0.0.0.0 so Docker can expose it
    app.run(host="0.0.0.0", port=5000)
