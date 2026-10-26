"""
predict.py
Command-line prediction: give 4 measurements, get the species.

Run:  python predict.py 5.1 3.5 1.4 0.2
"""
import sys

import joblib

MODEL_PATH = "model/iris_model.joblib"


def main():
    if len(sys.argv) != 5:
        print("Usage: python predict.py <sepal_length> <sepal_width> <petal_length> <petal_width>")
        sys.exit(1)

    features = [float(value) for value in sys.argv[1:]]

    bundle = joblib.load(MODEL_PATH)
    model, class_names = bundle["model"], bundle["class_names"]

    class_index = model.predict([features])[0]
    confidence = model.predict_proba([features])[0].max()
    print(f"Input: {features}")
    print(f"Predicted species: {class_names[class_index]} (confidence {confidence:.2f})")


if __name__ == "__main__":
    main()
