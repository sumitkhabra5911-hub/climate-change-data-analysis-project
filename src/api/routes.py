"""
FastAPI REST API Endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List

from src.api.schemas import (
    RiskPredictionRequest, RiskPredictionResponse,
    ForecastRequest, ForecastResponse,
    CountrySummary, CountryDetailResponse,
    HealthResponse
)
from src.api.service import model_service
from src.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Healthcheck endpoint returning system operational status and model availability."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        models_loaded=list(model_service.models.keys()),
        records_indexed=len(model_service.df) if model_service.df is not None else 0
    )


@api_router.post("/predict/risk", response_model=RiskPredictionResponse, tags=["Inference"])
def predict_climate_risk(payload: RiskPredictionRequest):
    """
    Evaluates real-time climate indicators against trained models
    to assess probability of high climate risk acceleration.
    """
    try:
        features_dict = payload.features.model_dump()
        result = model_service.predict_risk(features_dict, payload.model_name)
        return RiskPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_router.post("/predict/forecast", response_model=ForecastResponse, tags=["Inference"])
def predict_temperature_forecast(payload: ForecastRequest):
    """
    Generates multi-scenario temperature anomaly forecasts up to year 2100 with 95% confidence bounds.
    """
    try:
        result = model_service.predict_forecast(payload.target_year, payload.scenario)
        return ForecastResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_router.get("/analytics/countries", response_model=List[CountrySummary], tags=["Analytics"])
def list_countries():
    """Returns aggregated historical baseline statistics for all 30 monitored nations."""
    return model_service.get_countries_summary()


@api_router.get("/analytics/country/{country_name}", response_model=CountryDetailResponse, tags=["Analytics"])
def get_country(country_name: str):
    """Returns granular 1950–2023 historical records and trajectories for a specific nation."""
    try:
        detail = model_service.get_country_detail(country_name)
        return CountryDetailResponse(
            country=detail["country"],
            records=detail["records"],
            statistics=CountrySummary(**detail["statistics"]),
            status="success"
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Country '{country_name}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
