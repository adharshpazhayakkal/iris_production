"""
app.py
A REST API that serves the Iris model.

Endpoints
  GET  /health   -> is the service up, and which model is loaded?
  POST /predict  -> send 4 measurements as JSON, get the species back

Run locally:  python app.py   (then open http://localhost:5000/health)
"""
import os

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", "model/iris_model.joblib")

app = Flask(__name__)

# Load the model ONCE when the server starts, not on every request
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_names"]
CLASS_NAMES = bundle["class_names"]


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_accuracy": bundle["accuracy"],
        "trained_at": bundle["trained_at"],
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object in the request body"}), 400

    missing = [name for name in FEATURES if name not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = [float(data[name]) for name in FEATURES]
    except (TypeError, ValueError):
        return jsonify({"error": "All four fields must be numbers"}), 400

    probabilities = model.predict_proba([features])[0]
    best = int(probabilities.argmax())

    return jsonify({
        "prediction": CLASS_NAMES[best],
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities)
        },
    })


if __name__ == "__main__":
    # Development server only. In Docker we use gunicorn instead.
    app.run(host="0.0.0.0", port=5000, debug=False)
