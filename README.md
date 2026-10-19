# TerraPulse — Global Climate Intelligence & Prediction Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Pytest](https://img.shields.io/badge/Tests-9%20Passing-success?logo=pytest&logoColor=white)](https://pytest.org)

An enterprise-grade, end-to-end Climate Intelligence Platform combining machine learning risk classification, long-term temperature anomaly forecasting to 2050+, an interactive glassmorphic web dashboard, and a production-ready asynchronous REST API.

> 📖 **Exhaustive Code & Page Walkthrough**: For an in-depth line-by-line explanation of every module, algorithm, and UI tab, see [CODEBASE_EXPLANATION.md](file:///c:/Users/aviav/OneDrive/Documents/Desktop/climate%20change/CODEBASE_EXPLANATION.md).

---

## 🌟 Key Features & Capabilities

- **Enterprise REST API (FastAPI)**: Asynchronous, typed endpoints for real-time risk scoring, long-term forecasting, and country analytics with interactive Swagger docs (`/docs`).
- **Machine Learning Pipeline**:
  - **Risk Classification**: Logistic Regression (99.8% test accuracy), Random Forest, and Gradient Boosting to classify extreme warming scenarios (`High_Climate_Risk`).
  - **Forecasting Engine**: Linear Regression trend model projecting warming trajectories through 2050+ with 95% confidence intervals and multi-policy scenarios (BAU, Paris 1.5°C, Net-Zero).
- **Interactive Web Dashboard (TerraPulse)**:
  - 8-Chart interactive EDA deep dive (Chart.js).
  - Country Intelligence across 30 nations (1950–2023, 2,220 historical records).
  - Live AI scenario simulator with real-time risk factor attributions.
  - Data explorer with paginated search and one-click CSV export.
  - Dark & Light theme modes with persistence.
- **Production DevOps & QA**:
  - Pre-trained serialized models (`joblib`) in `outputs/models/`.
  - Comprehensive automated test suite with `pytest` covering API endpoints and model inferences.
  - Containerized with `Dockerfile` and `docker-compose.yml`.

---

## 🏗️ Architecture & Directory Structure

```text
climate change/
├── data/
│   ├── climate_data_raw.csv         # Generated raw climate records (30 countries, 1950-2023)
│   └── climate_data_cleaned.csv     # Cleaned, standardized ML-ready dataset
├── frontend/
│   ├── index.html                   # Glassmorphic semantic HTML5 dashboard
│   └── assets/
│       ├── css/styles.css           # Premium Vanilla CSS styling & themes
│       └── js/
│           ├── app.js               # Application logic, charts, and API connector
│           └── climate_data.js      # Bundled historical data & client fallback
├── notebooks/
│   └── 02_climate_analysis_and_prediction.ipynb  # Comprehensive Jupyter notebook
├── outputs/
│   ├── figures/                     # 12 high-resolution EDA and model evaluation plots
│   └── models/                      # Serialized ML models & metadata
│       ├── logistic_regression.joblib
│       ├── random_forest.joblib
│       ├── gradient_boosting.joblib
│       ├── temperature_forecaster.joblib
│       └── model_metadata.json
├── src/
│   ├── api/                         # Production FastAPI application
│   │   ├── app.py                   # App factory, lifespan, and middleware
│   │   ├── routes.py                # REST API routers
│   │   ├── schemas.py               # Pydantic V2 input/output schemas
│   │   └── service.py               # ML inference & analytical services
│   ├── config.py                    # Enterprise configuration settings
│   ├── data_generation.py           # Synthetic dataset generator
│   ├── data_cleaning.py             # Cleaning, validation, and outlier handling
│   ├── eda.py                       # Exploratory Data Analysis & visualizer
│   ├── modeling.py                  # ML training, cross-validation, and metrics
│   └── forecasting.py               # Linear anomaly forecasting to 2050
├── tests/
│   ├── test_api.py                  # API endpoint integration tests
│   └── test_models.py               # Data pipeline & model validation tests
├── Dockerfile                       # Multi-stage production container
├── docker-compose.yml               # Container orchestration
├── main.py                          # End-to-end data pipeline execution
├── run_server.py                    # Production Uvicorn server launcher
└── requirements.txt                 # Project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone or navigate to the directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
Execute the end-to-end pipeline to generate the dataset, generate the 12 figures, and serialize the models:
```bash
python main.py
```

### 3. Start the Production REST API & Web Dashboard
Launch the unified FastAPI server:
```bash
python run_server.py
```
- **Web Dashboard**: [http://localhost:8008](http://localhost:8008)
- **Interactive Swagger API Docs**: [http://localhost:8008/docs](http://localhost:8008/docs)
- **Alternative ReDoc**: [http://localhost:8008/redoc](http://localhost:8008/redoc)

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Service operational health and loaded models |
| `POST` | `/api/v1/predict/risk` | Real-time climate risk scoring and feature attribution |
| `POST` | `/api/v1/predict/forecast` | Multi-scenario anomaly projections up to 2100 |
| `GET` | `/api/v1/analytics/countries` | Aggregated metrics for all 30 countries |
| `GET` | `/api/v1/analytics/country/{name}` | 1950–2023 historical records for a country |

#### Example: Risk Prediction Request
```bash
curl -X POST "http://localhost:8008/api/v1/predict/risk" \
     -H "Content-Type: application/json" \
     -d '{
       "model_name": "Logistic Regression",
       "features": {
         "Year": 2024,
         "CO2_Emissions_MT": 1250.0,
         "CO2_per_Capita": 14.5,
         "Cumulative_CO2_MT": 45000.0,
         "Sea_Level_Rise_mm": 105.4,
         "Precipitation_mm": 850.0,
         "Humidity_pct": 65.2,
         "Population": 85000000.0,
         "Country_Baseline_Temp": 14.5,
         "Temp_Change_YoY": 0.25
       }
     }'
```

---

## 🧪 Automated Testing
Run the comprehensive test suite with `pytest`:
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment
Build and run the containerized service:
```bash
docker-compose up --build -d
```
Stop the container:
```bash
docker-compose down
```
