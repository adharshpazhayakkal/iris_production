"""
train.py
Trains a small classifier on the Iris dataset and saves it to disk.

Run:  python train.py
Output: model/iris_model.joblib
"""
import os
import sys
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.joblib")
MIN_ACCURACY = 0.90  # quality gate: CI fails if the model is worse than this


def main():
    # 1. Load data (150 flowers, 4 measurements each, 3 species)
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Hold back 20% of the data to test the model honestly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling + model in ONE pipeline, so serving applies the same scaling
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    model.fit(X_train, y_train)

    # 4. Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.3f}")

    # 5. Quality gate: refuse to save a bad model (non-zero exit fails CI)
    if accuracy < MIN_ACCURACY:
        print(f"FAILED: accuracy {accuracy:.3f} is below {MIN_ACCURACY}")
        sys.exit(1)

    # 6. Save the model together with the information needed to use it
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "class_names": list(iris.target_names),
        "accuracy": round(float(accuracy), 4),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
