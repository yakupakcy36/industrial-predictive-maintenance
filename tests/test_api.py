"""Test suite for FastAPI endpoints including RUL, XAI /predict/explain, and Mahalanobis drift."""

import pytest
from fastapi.testclient import TestClient
from api.app import app
from src.dataset import generate_industrial_telemetry


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "optimal_cost_threshold" in data


def test_predict_single_with_rul(client):
    payload = {
        "air_temperature_k": 300.2,
        "process_temperature_k": 310.5,
        "rotational_speed_rpm": 1540.0,
        "torque_nm": 39.8,
        "tool_wear_min": 45.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["estimated_rul_hours"] > 0.0
    assert data["alert_level"] in ["NORMAL", "WARNING", "CRITICAL"]


def test_predict_explain_xai_endpoint(client):
    payload = {
        "air_temperature_k": 303.0,
        "process_temperature_k": 308.0,
        "rotational_speed_rpm": 1260.0,
        "torque_nm": 66.0,
        "tool_wear_min": 210.0,
    }
    response = client.post("/predict/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "engineered_telemetry" in data
    assert "local_sensor_attributions" in data
    assert "counterfactual_recommendation" in data


def test_predict_batch_endpoint(client):
    payload = {
        "readings": [
            {
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500.0,
                "torque_nm": 40.0,
                "tool_wear_min": 20.0,
            },
            {
                "air_temperature_k": 304.0,
                "process_temperature_k": 308.0,
                "rotational_speed_rpm": 1280.0,
                "torque_nm": 65.0,
                "tool_wear_min": 220.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 2


def test_drift_evaluate_endpoint(client):
    sample_df = generate_industrial_telemetry(n_samples=25, seed=123)
    readings = sample_df[[
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
        "tool_wear_min",
    ]].to_dict(orient="records")

    response = client.post("/drift/evaluate", json={"readings": readings})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "multivariate_mahalanobis_mean" in data
    assert "multivariate_ood_fraction" in data
