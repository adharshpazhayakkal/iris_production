"""
client.py
Calls the running API exactly like another application would.
Uses only the Python standard library, so it works on any laptop.

Run (while the API is running):  python client.py
"""
import json
import urllib.error
import urllib.request

URL = "http://localhost:5000/predict"

flowers = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
    {"sepal_length": 5.1, "sepal_width": 3.5},  # bad request: two fields missing
]

for flower in flowers:
    request = urllib.request.Request(
        URL,
        data=json.dumps(flower).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            print(response.status, json.loads(response.read()))
    except urllib.error.HTTPError as err:
        print(err.code, json.loads(err.read()))
