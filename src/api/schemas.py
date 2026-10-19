"""
Pydantic V2 Request & Response Data Schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClimateFeaturesInput(BaseModel):
    Year: int = Field(..., ge=1950, le=2100, description="Observation year", json_schema_extra={"example": 2024})
    CO2_Emissions_MT: float = Field(..., ge=0.0, description="Annual CO2 emissions in Metric Megatons", json_schema_extra={"example": 1250.5})
    CO2_per_Capita: float = Field(..., ge=0.0, description="CO2 emissions per capita in tons", json_schema_extra={"example": 14.2})
    Cumulative_CO2_MT: float = Field(..., ge=0.0, description="Historical cumulative CO2 emissions", json_schema_extra={"example": 45000.0})
    Sea_Level_Rise_mm: float = Field(..., ge=-50.0, description="Global sea level rise relative to baseline (mm)", json_schema_extra={"example": 105.4})
    Precipitation_mm: float = Field(..., ge=0.0, description="Annual mean precipitation (mm)", json_schema_extra={"example": 850.0})
    Humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Average relative humidity (%)", json_schema_extra={"example": 65.2})
    Population: float = Field(..., ge=1000.0, description="National population", json_schema_extra={"example": 85000000.0})
    Country_Baseline_Temp: float = Field(..., description="Historic pre-industrial baseline temp (Celsius)", json_schema_extra={"example": 14.5})
    Temp_Change_YoY: float = Field(..., description="Year-over-Year temperature change (Celsius)", json_schema_extra={"example": 0.25})


class RiskPredictionRequest(BaseModel):
    model_name: Optional[str] = Field("Logistic Regression", description="Model engine: Logistic Regression, Random Forest, Gradient Boosting")
    features: ClimateFeaturesInput


class RiskFactorContribution(BaseModel):
    feature: str
    impact_score: float
    direction: str  # 'increases_risk' | 'decreases_risk'


class RiskPredictionResponse(BaseModel):
    model_used: str
    risk_prediction: int  # 0: Normal, 1: High Risk
    risk_label: str  # 'High Climate Risk Acceleration' | 'Normal / Controlled Risk'
    risk_probability: float
    confidence_percentage: float
    top_risk_factors: List[RiskFactorContribution]
    status: str = "success"


class ForecastRequest(BaseModel):
    target_year: int = Field(2050, ge=2024, le=2100, description="Target forecast year")
    scenario: Optional[str] = Field("business_as_usual", description="Scenario: business_as_usual, paris_agreement, net_zero")


class ForecastResponse(BaseModel):
    target_year: int
    scenario: str
    projected_anomaly_celsius: float
    lower_bound_95: float
    upper_bound_95: float
    estimated_sea_level_rise_mm: float
    warming_rate_celsius_per_decade: float
    status: str = "success"


class CountrySummary(BaseModel):
    country: str
    mean_temp_anomaly: float
    max_temp_anomaly: float
    latest_temp_anomaly: float
    latest_co2_mt: float
    cumulative_co2_mt: float
    high_risk_years_count: int
    records_count: int


class CountryDetailResponse(BaseModel):
    country: str
    records: List[Dict[str, Any]]
    statistics: CountrySummary
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]
    records_indexed: int
