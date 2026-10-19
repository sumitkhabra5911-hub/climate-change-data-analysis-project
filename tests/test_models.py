"""
Unit tests for data pipeline and model validation.
"""
import pytest
import pandas as pd
import numpy as np
from src.modeling import prepare_data, FEATURE_COLS
from src.api.service import model_service


def test_feature_columns_loaded():
    assert len(FEATURE_COLS) == 10
    assert model_service.df is not None
    for col in FEATURE_COLS:
        assert col in model_service.df.columns


def test_models_exist():
    assert "Logistic Regression" in model_service.models
    assert "Random Forest" in model_service.models
    assert "Gradient Boosting" in model_service.models
    assert "model" in model_service.forecaster_bundle


def test_model_inference_consistency():
    sample = {
        "Year": 2020,
        "CO2_Emissions_MT": 5000.0,
        "CO2_per_Capita": 10.0,
        "Cumulative_CO2_MT": 150000.0,
        "Sea_Level_Rise_mm": 90.0,
        "Precipitation_mm": 800.0,
        "Humidity_pct": 65.0,
        "Population": 50000000.0,
        "Country_Baseline_Temp": 15.0,
        "Temp_Change_YoY": 0.2
    }
    for model_name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
        res = model_service.predict_risk(sample, model_name=model_name)
        assert 0.0 <= res["risk_probability"] <= 1.0
        assert res["risk_prediction"] in [0, 1]
