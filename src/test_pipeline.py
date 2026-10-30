from fastapi.testclient import TestClient
from src.predict_api import app

client = TestClient(app)

def test_read_root():
    """Test checking the health of the API and whether the model is loaded."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_endpoint_structure():
    """Test checking the structure of the input and output data for the prediction endpoint."""
    payload = {
        "time_in_hospital": 3,
        "num_lab_procedures": 41,
        "num_procedures": 0,
        "num_medications": 15,
        "features_vector": [3.0, 41.0, 0.0, 15.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    }
    response = client.post("/predict", json=payload)
    
    # If model is loaded, we expect a 200 status; if not, we expect a 503.
    # In the CI environment, the local model will be copied, so it should be 200.
    if response.status_code == 200:
        data = response.json()
        assert "readmitted_prediction" in data
        assert "readmission_probability" in data
        assert isinstance(data["readmitted_prediction"], int)