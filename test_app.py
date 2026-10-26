"""
test_app.py
Automatic checks that GitHub Actions runs on every push.

Run:  pytest -v      (train the model first: python train.py)
"""
import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predicts_setosa(client):
    flower = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=flower)
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "setosa"


def test_predicts_virginica(client):
    flower = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
    response = client.post("/predict", json=flower)
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "virginica"


def test_probabilities_sum_to_one(client):
    flower = {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}
    probs = client.post("/predict", json=flower).get_json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_missing_field_returns_400(client):
    response = client.post("/predict", json={"sepal_length": 5.1})
    assert response.status_code == 400


def test_text_instead_of_number_returns_400(client):
    flower = {"sepal_length": "big", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=flower)
    assert response.status_code == 400


def test_no_json_returns_400(client):
    response = client.post("/predict", data="hello")
    assert response.status_code == 400
