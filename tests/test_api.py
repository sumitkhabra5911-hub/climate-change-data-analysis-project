"""
Integration Tests for REST API Endpoints using FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Logistic Regression" in data["models_loaded"]
    assert data["records_indexed"] > 0


def test_predict_risk_normal():
    payload = {
        "model_name": "Logistic Regression",
        "features": {
            "Year": 1960,
            "CO2_Emissions_MT": 150.0,
            "CO2_per_Capita": 4.5,
            "Cumulative_CO2_MT": 2500.0,
            "Sea_Level_Rise_mm": 5.0,
            "Precipitation_mm": 800.0,
            "Humidity_pct": 60.0,
            "Population": 30000000.0,
            "Country_Baseline_Temp": 14.0,
            "Temp_Change_YoY": 0.05
        }
    }
    response = client.post("/api/v1/predict/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_prediction" in data
    assert "risk_probability" in data
    assert len(data["top_risk_factors"]) > 0


def test_predict_risk_high():
    payload = {
        "model_name": "Logistic Regression",
        "features": {
            "Year": 2024,
            "CO2_Emissions_MT": 11500.0,
            "CO2_per_Capita": 16.5,
            "Cumulative_CO2_MT": 500000.0,
            "Sea_Level_Rise_mm": 110.0,
            "Precipitation_mm": 950.0,
            "Humidity_pct": 75.0,
            "Population": 1400000000.0,
            "Country_Baseline_Temp": 18.0,
            "Temp_Change_YoY": 0.65
        }
    }
    response = client.post("/api/v1/predict/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_prediction"] == 1
    assert data["risk_label"] == "High Climate Risk Acceleration"


def test_forecast_projections():
    payload = {"target_year": 2050, "scenario": "business_as_usual"}
    response = client.post("/api/v1/predict/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_year"] == 2050
    assert data["projected_anomaly_celsius"] > 3.0
    assert data["upper_bound_95"] > data["lower_bound_95"]


def test_countries_analytics():
    response = client.get("/api/v1/analytics/countries")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 30
    assert any(c["country"] == "United States" for c in data)


def test_country_detail():
    response = client.get("/api/v1/analytics/country/India")
    assert response.status_code == 200
    data = response.json()
    assert data["country"] == "India"
    assert len(data["records"]) == 74
