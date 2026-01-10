from flask import Flask, request, jsonify
from pathlib import Path
import torch
import numpy as np
import joblib
import pandas as pd

from models.binary import BinaryClassifier
from models.naive import NaiveZeroClassifier

import joblib

app = Flask(__name__)

# Expected raw fields (may be used to build a feature vector). Note: training
# drops `id` and `host_id`
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
    'reviews_per_month',
    'total_bookings'
]

# feats_ex =  {
# "id": 30419466,
# "features": [
# 0.86, // host_response_rate
# 1.0, // host_acceptance_rate
# 1, // host_is_superhost
# 20, // host_listings_count
# 24, // host_total_listings_count
# 2, // host_verifications
# 37.97251, // latitude
# 23.72772, // longitude
# 10, // accommodates
# 3.0, // bathrooms
# 4.0, // bedrooms
# 7.0, // beds
# 432.0, // price
# 181, // number_of_reviews
# 1, // instant_bookable
# 12, // calculated_host_listings_count
# 2.51, // reviews_per_month
# 5 // total_bookings
# ]
# }


# ----- Model loading -----
ROOT = Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = ROOT / "saved_models"
SCALER_PATH = SAVED_MODELS_DIR / "scaler.joblib"
_device = torch.device("cpu")

# Scaler (saved during training with joblib.dump)
SCALER_PATH = SAVED_MODELS_DIR / "scaler.joblib"
_scaler = None


def _load_binary_model(path: Path):
    # Load state dict first to infer input dimension
    state = torch.load(path, map_location=_device)
    # Find first 2D weight tensor (Linear weight) to infer input dim
    input_dim = None
    for k, v in state.items():
        if hasattr(v, 'ndim') and v.ndim == 2:
            # shape = (out_features, in_features)
            input_dim = v.shape[1]
            break

    if input_dim is None:
        raise RuntimeError("Could not infer input dimension from saved binary model")

    model = BinaryClassifier(input_dim)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    return model, input_dim


def _load_naive_model(path: Path):
    model = NaiveZeroClassifier()
    try:
        state = torch.load(path, map_location=_device)
        model.load_state_dict(state)
    except Exception:
        print("EXCEPTION")
        # naive model may have empty state dict; ignore
        pass
    model.to(_device)
    model.eval()
    return model


# Try to load models at import time; if missing, endpoints will report error.
_binary_model = None
_binary_input_dim = None
_naive_model = None

try:
    path = SAVED_MODELS_DIR / "binary_classifier_model.pth"
    print(f"Loading binary model from: {path} (exists={path.exists()})")
    _binary_model, _binary_input_dim = _load_binary_model(path)
    print("Binary model loaded, input_dim=", _binary_input_dim)
except Exception as e:
    print("Failed loading binary model:", repr(e))
    _binary_model = None

try:
    path = SAVED_MODELS_DIR / "naive_classifier_model.pth"
    print(f"Loading naive model from: {path} (exists={path.exists()})")
    _naive_model = _load_naive_model(path)
    print("Naive model loaded")

except Exception:
    _naive_model = None

try:
    if SCALER_PATH.exists():
        print(f"Loading scaler from: {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded")
    else:
        print(f"No scaler found at: {SCALER_PATH}")
except Exception as e:
    print("Failed loading scaler:", repr(e))
    _scaler = None


def _prepare_features(data: dict, expected_dim: int):
    """Return a numpy array shaped (1, D) for model input.

    Accepts either:
    - data['features'] : list/iterable of numbers
    - or key-value mapping containing REQUIRED_FIELDS (will skip id/host_id)
    """
    if 'features' in data:
        feats = np.asarray(data['features'], dtype=float)
    else:
        # Build vector from REQUIRED_FIELDS order (skip id and host_id)
        vec = []
        for f in REQUIRED_FIELDS:
            if f in ('id', 'host_id', 'listing_id'):
                continue
            if f in data:
                vec.append(data[f])
        feats = np.asarray(vec, dtype=float)

    if feats.ndim == 1:
        feats = feats.reshape(1, -1)

    # Apply scaler if available (scaler expects shape (n_samples, n_features))
    if _scaler is not None:
        try:
            # If scaler was fitted with feature names, provide a DataFrame
            # so sklearn won't warn about missing feature names when
            # transforming a numpy array.
            if hasattr(_scaler, "feature_names_in_"):
                if isinstance(feats, np.ndarray):
                    if feats.ndim == 1:
                        feats = feats.reshape(1, -1)
                    if feats.shape[1] == len(_scaler.feature_names_in_):
                        feats = pd.DataFrame(feats, columns=_scaler.feature_names_in_)

            feats = _scaler.transform(feats)
        except Exception as e:
            raise ValueError(f"Scaler transform failed: {e}")

    if feats.shape[1] != expected_dim:
        raise ValueError(f"Input features length {feats.shape[1]} does not match model expected {expected_dim}")

    return feats


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict/base", methods=["POST"])
def predict_base():
    if _naive_model is None:
        return jsonify({"error": "Naive model not available on server"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    try:
        # Naive model does not require a specific input dim, but we still expect a 2D array
        if 'features' in data:
            feats = np.asarray(data['features'], dtype=float)
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
        else:
            # Build from REQUIRED_FIELDS (skip id/host_id)
            vec = [data[f] for f in REQUIRED_FIELDS if f not in ('id', 'host_id') and f in data]
            if 'total_bookings' in data:
                vec.append(data['total_bookings'])
            feats = np.asarray(vec, dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    x = torch.tensor(feats, dtype=torch.float32, device=_device)
    with torch.no_grad():
        probs = _naive_model(x).cpu().numpy().ravel().tolist()

    result = {
        "prediction": [int(p > 0.5) for p in probs],
        "probability": probs
    }
    if 'id' in data:
        result['id'] = data['id']

    return jsonify(result), 200


@app.route("/predict/binary", methods=["POST"])
def predict_binary():
    if _binary_model is None:
        return jsonify({"error": "Binary model not available on server"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    try:
        feats = _prepare_features(data, _binary_input_dim)
    except ValueError as e:
        return jsonify({"error": str(e), "hint": "Provide numeric 'features' list of correct length."}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    x = torch.tensor(feats, dtype=torch.float32, device=_device)
    with torch.no_grad():
        probs = _binary_model(x).cpu().numpy().ravel()

    preds = (probs > 0.5).astype(int).tolist()  # sigmoid at the end so threshold at 0.5

    result = {#todo sukcesja modeli
        # todo strojenie hiperparametrow
        # porownanie modeli jest
        # roznice modeli - opisac
        # set seed
        # odniesienie sie do kryteriow suckesu w.z. z czym
        # test AB
        # "model": "binary_classifier",
        "prediction": preds,
        "probability": probs.tolist()
    }
    if 'id' in data:
        result['id'] = data['id']

    return jsonify(result), 200


@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({"error": "Use /predict/base or /predict/binary endpoints"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
