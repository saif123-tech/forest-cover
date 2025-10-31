from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Forest Cover Prediction API", "status": "running"}

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    json_data = response.json()
    assert "model_type" in json_data
    assert json_data["model_type"] == "Random Forest Classifier"
    assert json_data["features_required"] == 54

def test_predict_valid():
    # Provide a valid input with 54 features (all zeros)
    features = [0.0] * 54
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert "cover_type_name" in json_data

def test_predict_invalid_length():
    # Provide invalid input with wrong number of features
    features = [0.0] * 10
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 400
    assert "Input must have exactly 54 features" in response.json()["detail"]

def test_predict_no_model(monkeypatch):
    # Temporarily set model to None to test error handling
    from app import model
    monkeypatch.setattr("app.model", None)
    features = [0.0] * 54
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 500
    assert "Model not loaded" in response.json()["detail"]
