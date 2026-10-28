
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

PASSENGER = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S",
}


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post("/predict", json=PASSENGER)
    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_invalid_passenger():
    response = client.post("/predict", json={})
    assert response.status_code == 422