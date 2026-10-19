# TerraPulse — Comprehensive Codebase & Architecture Walkthrough

This document provides an exhaustive, line-by-line and section-by-section breakdown of every file, module, script, endpoint, mathematical equation, and user interface component across the entire **TerraPulse Climate Intelligence & Prediction Platform**.

---

## 📑 Table of Contents
1. [High-Level Architecture & Data Flow](#1-high-level-architecture--data-flow)
2. [Root Execution & Configuration Scripts](#2-root-execution--configuration-scripts)
   - [`main.py`](#mainpy--master-pipeline-orchestrator)
   - [`run_server.py`](#run_serverpy--production-uvicorn-server)
   - [`src/config.py`](#srcconfigpy--enterprise-settings)
3. [Core Data Science & Machine Learning Modules (`src/`)](#3-core-data-science--machine-learning-modules-src)
   - [`src/data_generation.py`](#srcdata_generationpy--synthetic-climate-generator)
   - [`src/data_cleaning.py`](#srcdata_cleaningpy--imputation--feature-engineering)
   - [`src/eda.py`](#srcedapy--exploratory-data-analysis--visualizations)
   - [`src/modeling.py`](#srcmodelingpy--ml-classification-benchmark--serialization)
   - [`src/forecasting.py`](#srcforecastingpy--linear-anomaly-forecasting-to-2050)
4. [Production REST API (`src/api/`)](#4-production-rest-api-srcapi)
   - [`src/api/app.py`](#srcapiapppy--fastapi-application-factory--lifespan)
   - [`src/api/schemas.py`](#srcapischemaspy--pydantic-v2-data-contracts)
   - [`src/api/service.py`](#srcapiservicepy--model-inference--analytical-engine)
   - [`src/api/routes.py`](#srcapiroutespy--rest-api-endpoints)
5. [Data Export & Synchronization Scripts (`scripts/`)](#5-data-export--synchronization-scripts-scripts)
   - [`scripts/export_frontend_data.py`](#scriptsexport_frontend_datapy--frontend-bundle-builder)
   - [`scripts/build_notebook.py`](#scriptsbuild_notebookpy--jupyter-notebook-generator)
6. [Interactive Frontend Dashboard (`frontend/`)](#6-interactive-frontend-dashboard-frontend)
   - [`frontend/index.html` (Every Page & Tab Section)](#frontendindexhtml--glassmorphic-dashboard)
   - [`frontend/assets/js/app.js`](#frontendassetsjsappjs--client-logic--api-connectors)
   - [`frontend/assets/css/styles.css`](#frontendassetscssstylescss--glassmorphic-design-system)
7. [Automated Testing Suite (`tests/`)](#7-automated-testing-suite-tests)
   - [`tests/test_api.py`](#teststest_apipy--fastapi-endpoint-integration-tests)
   - [`tests/test_models.py`](#teststest_modelspy--model-validation--integrity-tests)
8. [DevOps & Containerization](#8-devops--containerization)
   - [`Dockerfile` & `docker-compose.yml`](#dockerfile--docker-composeyml)

---

## 1. High-Level Architecture & Data Flow

TerraPulse operates as an end-to-end, decoupled yet synchronized data science and web platform:

```
[ Synthetic Generator (src/data_generation.py) ]
                   │
                   ▼  (2,220 rows across 30 countries, 1950-2023)
        [ data/climate_data_raw.csv ]
                   │
                   ▼  (Country-specific median imputation + Feature Engineering)
[ Data Cleaning & Feature Pipeline (src/data_cleaning.py) ]
                   │
                   ▼
       [ data/climate_data_cleaned.csv ]
      ┌────────────┼─────────────────────────┐
      │            │                         │
      ▼            ▼                         ▼
[ EDA Visuals ] [ Supervised Classifiers ] [ Linear Forecaster ]
(src/eda.py)    (src/modeling.py)          (src/forecasting.py)
  12 Figures      3 Classifiers Benchmarked  Linear Regression to 2050
  outputs/figs/   outputs/models/*.joblib    outputs/models/*.joblib
                   │                         │
                   └──────────┬──────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
[ FastAPI Backend Engine ]         [ Frontend Data Bundle ]
(src/api/service.py & routes.py)   (scripts/export_frontend_data.py)
Port 8008 REST API / Swagger docs  frontend/assets/js/climate_data.js
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                [ Interactive Glassmorphic UI ]
                  (frontend/index.html & app.js)
```

---

## 2. Root Execution & Configuration Scripts

### `main.py` — Master Pipeline Orchestrator
- **Purpose**: A single entry point that reproduces the entire project from raw data synthesis to trained models, figures, frontend bundles, and Jupyter notebooks.
- **Line-by-Line Breakdown**:
  - `lines 1-22`: Imports OS, pipeline sub-functions (`generate_climate_dataset`, `clean_climate_data`, `engineer_features`, `run_all_eda`, `prepare_data`, `train_and_evaluate_models`, `save_models`, `forecast_temperature_to_2050`, `export_data`, `build_notebook`).
  - `lines 24-28`: `run_pipeline()` entry point with banner logging.
  - `lines 29-34 (Step 1/6)`: Invokes `generate_climate_dataset()` and saves raw records to `data/climate_data_raw.csv`.
  - `lines 35-42 (Step 2/6)`: Runs `clean_climate_data()` (imputes missing records with per-country medians) and `engineer_features()` (creates anomaly, moving averages, and binary risk classification target), saving to `data/climate_data_cleaned.csv`.
  - `lines 43-47 (Step 3/6)`: Calls `run_all_eda()` to output 8 publication-grade charts into `outputs/figures/`.
  - `lines 48-60 (Step 4/6)`: Splits features using stratified 80/20 train/test split, trains Logistic Regression, Random Forest, and Gradient Boosting, evaluates them with 5-fold CV, plots ROC curves, confusion matrices, and feature importances, and calls `save_models()` to serialize models into `outputs/models/`.
  - `lines 61-68 (Step 5/6)`: Computes global annual warming anomalies, fits a linear regression trend to 2050 with 95% confidence intervals, plots Chart 12, and saves `temperature_forecaster.joblib`.
  - `lines 69-74 (Step 6/6)`: Invokes `export_data()` to update the client-side JavaScript bundle and `build_notebook()` to regenerate `02_climate_analysis_and_prediction.ipynb`.
  - `lines 77-79`: Standard Python `if __name__ == "__main__":` idiom.

---

### `run_server.py` — Production Uvicorn Server
- **Purpose**: Launches the asynchronous FastAPI server with production configurations.
- **Line-by-Line Breakdown**:
  - `lines 4-5`: Imports `uvicorn` and the centralized `settings` object from `src.config`.
  - `lines 7-10`: Prints welcoming URLs for the dashboard and the interactive Swagger documentation.
  - `lines 11-17`: Calls `uvicorn.run("src.api.app:app", host=settings.HOST, port=settings.PORT, reload=False, workers=1)`. Binds directly to `0.0.0.0:8008`.

---

### `src/config.py` — Enterprise Settings
- **Purpose**: Pydantic-settings configuration for environment variables and paths.
- **Line-by-Line Breakdown**:
  - `lines 5-6`: Imports `BaseSettings` and `SettingsConfigDict` from `pydantic_settings`.
  - `lines 7-14`: Defines the `Settings` class:
    - `PROJECT_NAME`: `"TerraPulse Climate Intelligence API"`
    - `VERSION`: `"1.0.0"`
    - `API_V1_STR`: `"/api/v1"` (prefixed API path)
    - `HOST`: `"0.0.0.0"` (listens on all network interfaces)
    - `PORT`: `8008` (avoids port collisions with local port 8000 services)
    - `CORS_ORIGINS`: `["*"]` (allows cross-origin requests)
  - `lines 16-18`: Resolves absolute filesystem paths:
    - `BASE_DIR`: Project root directory.
    - `DATA_PATH`: Cleaned dataset path (`data/climate_data_cleaned.csv`).
    - `MODELS_DIR`: Directory where serialized joblib models reside (`outputs/models`).
  - `line 20`: `model_config = SettingsConfigDict(case_sensitive=True)` ensures strict Pydantic V2 compliance with zero deprecation warnings.
  - `line 22`: Instantiates the singleton `settings = Settings()`.

---

## 3. Core Data Science & Machine Learning Modules (`src/`)

### `src/data_generation.py` — Synthetic Climate Generator
- **Purpose**: Generates realistic, physically-sound climate observations for 30 nations across 74 years (1950–2023 = 2,220 rows) with realistic non-linearities and real-world data collection flaws.
- **Line-by-Line Breakdown**:
  - `lines 12-43`: `COUNTRIES_CONFIG` dictionary defining physical baselines for 30 nations across 6 continents:
    - `base_temp`: Pre-industrial mean temperature (e.g., -5.1°C for Russia, 26.8°C for Indonesia).
    - `warming_rate`: Regional warming multiplier (polar amplification in Canada: 1.60x, Russia: 1.55x vs. tropical 0.90x).
    - `base_pop` & `pop_growth`: 1950 population and historical demographic expansion rates.
    - `base_co2` & `co2_growth`: Industrial scale and emission trajectories.
    - `precip` & `humidity`: Natural precipitation and baseline relative humidity.
  - `lines 46-54`: `generate_climate_dataset()` signature with parameters for year bounds, random seed (42 for reproducibility), and missingness rate (3%).
  - `lines 55-66`: Iterates over each country and step index $t = \text{year} - 1950$.
  - `lines 68-75 (Warming Dynamics)`:
    - Piecewise acceleration formula:
      $$\text{accel\_factor} = \begin{cases} 0.008 \cdot t^{1.35} & t < 30 \text{ (pre-1980)} \\ 0.008 \cdot 30^{1.35} + 0.032 \cdot (t - 30)^{1.15} & t \ge 30 \text{ (post-1980 surge)} \end{cases}$$
    - Adds multi-frequency periodic cycles simulating ENSO (5.2-year cycle) and the Solar Magnetic Cycle (11-year cycle):
      $$\text{enso\_cycle} = 0.22 \sin\left(\frac{2\pi t}{5.2}\right) + 0.15 \cos\left(\frac{2\pi t}{11.0}\right)$$
    - Adds Gaussian noise $\mathcal{N}(0, 0.28^2)$.
  - `lines 77-79 (Demographics)`: Exponential growth modulated by logistical slowing in the 21st century.
  - `lines 80-93 (CO2 Emissions)`:
    - Developed economies (USA, Germany, UK, France): Quadratic peaking followed by decarbonization declines post-2005.
    - Rapidly industrializing nations (China, India, Vietnam): Sigmoidal surge post-1990:
      $$\text{boost} = 1 + \frac{e^{(year - 1990)/10}}{1 + e^{(year - 1990)/10}} \cdot 3.5$$
    - Rest of the world: Steady exponential growth.
  - `lines 94-96 (Sea Level Rise)`: Quadratic cumulative equation reaching ~220mm:
    $$\text{sea\_level} = 0.9t + 0.028t^2 + \mathcal{N}(0, 2.5)$$
  - `lines 98-106 (Precipitation & Humidity)`: Interannual atmospheric volatility with Clausius-Clapeyron moisture dynamics.
  - `lines 118-124`: Injects ~3% synthetic missing values (`np.nan`) into observational columns to represent real-world missingness.
  - `lines 126-137`: Helper `save_dataset()` exports to CSV.

---

### `src/data_cleaning.py` — Imputation & Feature Engineering
- **Purpose**: Cleans data using country-specific median imputation and engineers domain-specific predictive indicators.
- **Line-by-Line Breakdown**:
  - `lines 12-40 (clean_climate_data)`:
    - Sorts records strictly by `["Country", "Year"]`.
    - Imputes missing numerical entries via `df.groupby("Country")[col].transform(lambda grp: grp.fillna(grp.median()))`. This prevents geographic skew (e.g., preventing Egypt's temperature from imputing Norway's missing winter readings).
    - Provides a global median fallback in case an entire country slice was null.
  - `lines 42-85 (engineer_features)`:
    1. `CO2_per_Capita`: Normalized metric tons per citizen:
       $$\text{CO2\_per\_Capita} = \frac{\text{CO2\_Emissions\_MT} \times 10^6}{\text{Population}}$$
    2. `Country_Baseline_Temp`: Pre-industrial reference period mean (1950–1980):
       $$\bar{T}_{\text{baseline}} = \frac{1}{N_{1950-80}} \sum_{t=1950}^{1980} T_t$$
    3. `Temp_Anomaly`: Local deviation: $\Delta T = T_t - \bar{T}_{\text{baseline}}$.
    4. `Temp_Change_YoY`: First difference in temperature: $T_t - T_{t-1}$.
    5. `Temp_5yr_MA`: 5-year rolling moving average to filter short-term climate noise.
    6. `Cumulative_CO2_MT`: Cumulative sum of country carbon output since 1950.
    7. `High_Climate_Risk` (Supervised Classification Target): Binary threshold indicator defined as local anomaly exceeding the 75th percentile of historical warming ($> +2.18^\circ\text{C}$):
       $$\text{High\_Climate\_Risk} = \begin{cases} 1 & \text{if } \text{Temp\_Anomaly} > Q_{0.75} \\ 0 & \text{otherwise} \end{cases}$$
  - `lines 88-97`: `process_and_save()` pipeline orchestrator.

---

### `src/eda.py` — Exploratory Data Analysis & Visualizations
- **Purpose**: Generates 8 publication-grade scientific charts saved as high-resolution (150 DPI) PNGs in `outputs/figures/`.
- **Chart Details**:
  - `plot_01_global_temp_trend`: Dual-axis bar and smoothed 5-year moving average curve revealing accelerating post-1980 warming.
  - `plot_02_top_emitters`: Horizontal bar chart ranking the top 10 cumulative gigaton emitters (USA, China, Russia, etc.).
  - `plot_03_correlation_heatmap`: Annotated Pearson correlation matrix identifying strong collinearity between CO2, sea-level rise, and temperature anomalies.
  - `plot_04_sea_level_rise`: Filled-area chart showing the quadratic curve of global sea-level expansion reaching 218mm.
  - `plot_05_temp_anomaly_decades`: Decadal boxplots (1950s to 2020s) illustrating the continuous rightward shift and widening variance of global temperatures.
  - `plot_06_co2_vs_temp_scatter`: Scatter plot of per-capita emissions vs. temperature anomalies colored by risk level.
  - `plot_07_precipitation_trends`: Dual-line chart of global annual precipitation volatility and moving average.
  - `plot_08_multi_country_trajectories`: Multi-line comparative chart comparing polar warming (Canada, Norway) with equatorial warming (Brazil, India).
  - `run_all_eda()`: Executes all 8 plotting functions sequentially.

---

### `src/modeling.py` — ML Classification Benchmark & Serialization
- **Purpose**: Prepares feature matrices, benchmarks 3 distinct classifiers, computes 5-fold cross-validation and test metrics, plots evaluation charts, and serializes artifacts.
- **Line-by-Line Breakdown**:
  - `lines 25-37`: Defines `FEATURE_COLS` (10 variables: `Year`, `CO2_Emissions_MT`, `CO2_per_Capita`, `Cumulative_CO2_MT`, `Sea_Level_Rise_mm`, `Precipitation_mm`, `Humidity_pct`, `Population`, `Country_Baseline_Temp`, `Temp_Change_YoY`) and `TARGET_COL` (`High_Climate_Risk`).
  - `lines 40-49`: `prepare_data()` performs a stratified 80/20 train/test split preserving target class balance.
  - `lines 51-65`: `get_models()` initializes:
    1. **Logistic Regression**: Encapsulated in a `StandardScaler()` pipeline to ensure uniform feature weight attribution ($L_2$ regularization, $C=1.0$).
    2. **Random Forest**: Ensemble of 150 decision trees, `max_depth=10`, `min_samples_split=4`.
    3. **Gradient Boosting**: Sequential ensemble of 120 estimators, `learning_rate=0.08`, `max_depth=5`.
  - `lines 67-114`: `train_and_evaluate_models()`:
    - Runs 5-Fold Stratified Cross-Validation on the training set to measure generalization stability (`CV_Accuracy_Mean` and `Std`).
    - Trains models on full training set and evaluates holdout metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - `lines 116-150`: `plot_model_evaluations()` generates:
    - Chart 09: Multi-model ROC curves with AUC annotations.
    - Chart 10: Side-by-side Confusion Matrices on the test set.
  - `lines 152-169`: `plot_feature_importance()` generates Chart 11, ranking feature Gini importances for Random Forest and Gradient Boosting.
  - `lines 172-198`: `save_models()` serializes:
    - `logistic_regression.joblib`
    - `random_forest.joblib`
    - `gradient_boosting.joblib`
    - `model_metadata.json` (version, training records, features, performance metrics).

---

### `src/forecasting.py` — Linear Anomaly Forecasting to 2050
- **Purpose**: Fits a linear trend model to historical anomalies and projects warming trajectories with 95% confidence intervals up to 2050+.
- **Line-by-Line Breakdown**:
  - `lines 20-28`: Aggregates global mean anomaly per year and fits `sklearn.linear_model.LinearRegression()`.
  - `lines 29-36`: Extracts linear slope ($+0.472^\circ\text{C}$ per decade) and computes residual standard deviation $\sigma_{\text{res}}$ on historical fit:
    $$\sigma_{\text{res}} = \sqrt{\frac{1}{N - 2} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
  - `lines 38-51`: Forecasts future years (2024–2050) and calculates 95% prediction intervals:
    $$\text{CI}_{95\%} = \hat{y} \pm 1.96 \cdot \sigma_{\text{res}}$$
  - `lines 53-107`: Plots Chart 12 (`12_temperature_forecast_2050.png`) showing the historical curve, linear fit, forecast trajectory, 95% confidence envelope, and the critical 1.5°C Paris Agreement limit.
  - `lines 14-25`: `save_forecaster()` bundles the model, slope, intercept, and residual standard deviation into `outputs/models/temperature_forecaster.joblib`.

---

## 4. Production REST API (`src/api/`)

### `src/api/app.py` — FastAPI Application Factory & Lifespan
- **Purpose**: Creates the production FastAPI instance, configures CORS, handles asynchronous application lifecycle, and serves the static frontend dashboard.
- **Line-by-Line Breakdown**:
  - `lines 9-20`: `@asynccontextmanager` defines the `lifespan` handler. On startup, it triggers `model_service.load_artifacts()` to load models and data into memory, logging system readiness.
  - `lines 23-34`: Instantiates `FastAPI` with metadata, Swagger (`/docs`), and ReDoc (`/redoc`) documentation endpoints.
  - `lines 36-42`: Configures `CORSMiddleware` to allow cross-origin requests.
  - `lines 45-47`: Mounts API routers from `src.api.routes` under `/api/v1`.
  - `lines 49-59`: Mounts the static directory (`frontend/`) and serves `frontend/index.html` at the root path (`/`).

---

### `src/api/schemas.py` — Pydantic V2 Data Contracts
- **Purpose**: Enforces strict input validation and serialized output schemas.
- **Contract Breakdown**:
  - `ClimateFeaturesInput`: Validates all 10 environmental indicators (bounds: `Year` 1950–2100, `Humidity_pct` 0–100%, positive bounds on CO2, population, etc.).
  - `RiskPredictionRequest`: Accepts `model_name` (optional, defaults to `"Logistic Regression"`) and `features` (`ClimateFeaturesInput`).
  - `RiskFactorContribution`: Structure for individual feature impact score and direction (`increases_risk` / `decreases_risk`).
  - `RiskPredictionResponse`: Returns model used, binary classification, descriptive label, probability, confidence percentage, and top 5 risk drivers.
  - `ForecastRequest`: Accepts `target_year` (2024–2100) and `scenario` (`business_as_usual`, `paris_agreement`, `net_zero`).
  - `ForecastResponse`: Returns target year, scenario, projected anomaly, 95% confidence intervals, estimated sea level rise, and decadal warming rate.
  - `CountrySummary` & `CountryDetailResponse`: Returns analytical aggregations and historical records for any country.
  - `HealthResponse`: System operational status, version, and loaded models.

---

### `src/api/service.py` — Model Inference & Analytical Engine
- **Purpose**: Business logic layer managing model inference, explainability attributions, multi-scenario projections, and country aggregations.
- **Key Methods**:
  - `load_artifacts()`: Reads `climate_data_cleaned.csv`, `model_metadata.json`, all three classification `.joblib` models, and the `temperature_forecaster.joblib` bundle.
  - `predict_risk(features_dict, model_name)`:
    - Builds a 1-row feature vector conforming to `FEATURE_COLS`.
    - Predicts binary risk label ($0$ or $1$) and probability $P(Y=1)$.
    - **Feature Attribution Scoring**:
      - For *Logistic Regression*: Computes normalized log-odds contribution $z_i \cdot \beta_i = \left(\frac{x_i - \mu_i}{\sigma_i}\right) \cdot w_i$.
      - For *Random Forest / Gradient Boosting*: Evaluates Gini feature importance contributions.
    - Sorts and returns top 5 risk factors with directionality.
  - `predict_forecast(target_year, scenario)`:
    - Computes baseline linear regression projection: $\hat{y} = \text{intercept} + \text{slope} \cdot \text{year}$.
    - **Scenario Adjustments**:
      - `paris_agreement`: Non-linear saturation targeting 1.5°C ceiling:
        $$\hat{y}_{\text{paris}} = \min\left(\hat{y}_{\text{bau}}, 1.45 + 0.05 \cdot (1 - e^{-\Delta t / 15})\right)$$
      - `net_zero`: Aggressive mitigation path stabilizing around 1.35°C:
        $$\hat{y}_{\text{netzero}} = \min\left(\hat{y}_{\text{bau}}, 1.30 + 0.15 \cdot (1 - e^{-\Delta t / 12})\right)$$
    - Calculates empirical sea-level expansion:
      $$\text{SLR}_{\text{mm}} = 105.0 + \Delta t \cdot \left(3.8 + (\hat{y} - 1.2) \cdot 1.5\right)$$
  - `get_countries_summary()`: Computes multi-year statistical metrics across all 30 countries.
  - `get_country_detail(country_name)`: Returns all 74 time-series records for an individual country.

---

### `src/api/routes.py` — REST API Endpoints
- **Purpose**: Maps HTTP verbs to `ModelService` operations:
  - `GET /api/v1/health`: Returns health status, loaded models, and indexed records.
  - `POST /api/v1/predict/risk`: Runs real-time risk classification and factor attributions.
  - `POST /api/v1/predict/forecast`: Calculates multi-scenario climate projections.
  - `GET /api/v1/analytics/countries`: Retrieves summary table for all 30 countries.
  - `GET /api/v1/analytics/country/{name}`: Retrieves time-series history for a selected nation.

---

## 5. Data Export & Synchronization Scripts (`scripts/`)

### `scripts/export_frontend_data.py` — Frontend Bundle Builder
- **Purpose**: Pre-calculates heavy analytics and serializes them into `frontend/assets/js/climate_data.js` as `window.CLIMATE_DATA`.
- **Pre-Calculated Payloads**:
  1. `global_annual`: Global mean temperature, anomaly, CO2, sea level, and precipitation per year.
  2. `country_profiles`: 1950 vs. 2023 temps, cumulative carbon emissions, and full 74-year historical records for all 30 nations.
  3. `top_emitters`: Cumulative gigaton ranking.
  4. `decades`: Five-number summary boxplot metrics (min, Q25, median, Q75, max) for every decade from the 1950s to 2020s.
  5. `correlation`: 7x7 Pearson correlation matrix.
  6. `ml_model`: Serialized scaler means, scales, logistic regression weights, intercept, and Random Forest feature importances for instant client-side offline execution.
  7. `forecast`: 2024–2050 multi-scenario data points and confidence intervals.
  8. `table_rows`: 2,220 compact rows formatted for instant client-side searching and CSV export.

---

### `scripts/build_notebook.py` — Jupyter Notebook Generator
- **Purpose**: Generates the complete, self-contained `notebooks/02_climate_analysis_and_prediction.ipynb` in valid Jupyter Notebook v4.5 JSON format without requiring heavy notebook dependencies during pipeline runs.
- **Included Sections**:
  - Overview and Library Setup
  - Synthetic Data Generation with Missingness Simulation
  - Country-Specific Median Imputation
  - Domain Feature Engineering
  - 8-Chart Publication EDA
  - Supervised Machine Learning Benchmark (Logistic Regression, Random Forest, Gradient Boosting)
  - Cross-Validation, ROC Curves, and Confusion Matrices
  - Time-Series Forecasting to 2050

---

## 6. Interactive Frontend Dashboard (`frontend/`)

### `frontend/index.html` — Glassmorphic Dashboard
Structured into 6 distinct navigation sections and interactive modules:

1. **Header & Live Status Indicators**:
   - Brand logo and platform identity.
   - Live REST API status pulse (`Production API: Online`).
   - Benchmark Model Accuracy badge (`95.95%`).
   - Theme toggle button (Dark / Light mode).
2. **Global KPI Metric Cards**:
   - `2023 Global Anomaly`: `+2.18°C` (▲ +218% vs baseline).
   - `Cumulative Global CO2`: `1.25 TMT` (Accelerating emissions).
   - `Mean Sea Level Rise`: `+218.4 mm` (Quadratic acceleration).
   - `ML Model Test Accuracy`: `95.95%` (ROC-AUC 0.991).
3. **Tab 1: Overview (`#section-overview`)**:
   - Global Temperature Anomaly Trajectory (1950–2023) line chart with 5-year moving average.
   - Global Sea Level Rise chart.
   - Top Historical Emitters horizontal bar chart.
4. **Tab 2: 8-Chart EDA Deep Dive (`#section-eda`)**:
   - Chart 1: Global Temperature Anomaly Trend bars with smoothing curve.
   - Chart 2: Top 10 Cumulative CO2 Emitters in gigatons.
   - Chart 3: Interactive Heatmap Correlation Table with dynamic color scale.
   - Chart 4: Cumulative Sea Level Rise Acceleration.
   - Chart 5: Decadal Temperature Distribution progression (1950s baseline to 2020s extremes).
   - Chart 6: CO2 per Capita vs. Temperature Anomaly scatter plot.
   - Chart 7: Annual Precipitation Trends and volatility.
   - Chart 8: Regional Warming Divergence (polar vs. tropical warming).
5. **Tab 3: Country Intelligence (`#section-country`)**:
   - Nation selector dropdown populated with all 30 countries.
   - Geographic summary cards: Baseline Temp, 1950 Temp, 2023 Temp, Current Anomaly, Cumulative CO2, Population.
   - Interactive dual-axis historical trend chart (1950–2023).
6. **Tab 4: AI Climate Risk Simulator (`#section-simulator`)**:
   - **Model Engine Dropdown**: Choose between `Logistic Regression`, `Random Forest`, and `Gradient Boosting`.
   - **Interactive Range Sliders**: Observation Year (1950–2050), Annual CO2, Population, Sea Level Rise, Precipitation, Humidity, Baseline Temp, YoY Acceleration.
   - **Live Prediction Verdict Card**: Real-time calculated risk probability and classification tag.
   - **Feature Impact Attributions**: Visual progress bars showing the top 5 factors increasing or decreasing climate risk.
   - **Model Benchmark Cards**: Comparative metrics for all 3 models.
7. **Tab 5: 2050 Forecast Simulator (`#section-forecast`)**:
   - Multi-pathway forecast chart displaying Historical Observations, Business-as-Usual, 95% Confidence Interval envelope, 1.5°C Paris Agreement limit, and Net-Zero mitigation curves.
   - Forecast target year slider (2024–2050) and Policy Pathway selector.
   - Projected Anomaly, Confidence Interval, and Estimated Sea Level Rise metric cards.
8. **Tab 6: Data Explorer & CSV Export (`#section-table`)**:
   - Search input (searches by Country name or Year).
   - Risk classification filter (`All`, `High Risk Only`, `Normal Only`).
   - One-click CSV Export button that downloads filtered rows directly to the user's browser.
   - Paginated table (15 records per page) with page navigation controls.

---

### `frontend/assets/js/app.js` — Client Logic & API Connectors
- **Global State**: Tracks active theme, current navigation tab, selected country, Chart.js instances, table pagination, and search filters.
- **REST API Integration**:
  - `checkApiHealth()`: Queries `/api/v1/health` on load to update the header status badge.
  - `runMLInference()`:
    - Formats environmental slider values into a typed JSON request.
    - Sends `POST /api/v1/predict/risk` to the live FastAPI backend.
    - If the API is online, it dynamically renders server-side inferences and exact feature attributions.
    - If offline, it smoothly falls back to client-side linear algebra calculations using weights from `window.CLIMATE_DATA`.
  - `updateForecastMilestones()`:
    - Queries `POST /api/v1/predict/forecast` to fetch server-side projections and sea level calculations.
- **Visualizations**: Configures 11 distinct Chart.js instances with responsive canvas resizing and dark/light color palette updates.
- **Table Controller**: Implements client-side fuzzy search, risk category filtering, pagination slicing, and CSV blob export.

---

### `frontend/assets/css/styles.css` — Glassmorphic Design System
- **CSS Custom Properties**: Curated HSL color palette:
  - `--bg-primary`, `--bg-secondary`, `--bg-card` (glassmorphic translucent surfaces with `backdrop-filter: blur(16px)`).
  - Accent colors: Coral (`#ef4444`), Amber (`#f59e0b`), Emerald (`#10b981`), Cyan (`#06b6d4`), Violet (`#8b5cf6`).
- **Typography**: Google Fonts integration using `Outfit` for bold headings, `Plus Jakarta Sans` for body text, and `JetBrains Mono` for data and metrics.
- **Layouts**: Responsive CSS Grid layouts (`.grid-2col`, `.kpi-grid`, `.simulator-container`) with mobile breakpoint media queries.
- **Animations**: Subtle hover lifts, glowing status pulse dots, and smooth cubic-bezier transitions.

---

## 7. Automated Testing Suite (`tests/`)

### `tests/test_api.py` — FastAPI Endpoint Integration Tests
Executes integration tests using Starlette's `TestClient`:
1. `test_health_check`: Asserts `/api/v1/health` returns status 200, healthy status, loaded models, and indexed records $>0$.
2. `test_predict_risk_normal`: Sends baseline historical conditions and asserts normal risk prediction ($0$).
3. `test_predict_risk_high`: Sends severe warming levers (high CO2, high sea level rise) and asserts `risk_prediction == 1` and `"High Climate Risk Acceleration"`.
4. `test_forecast_projections`: Asserts 2050 forecast returns valid projections and proper confidence interval order (`upper > lower`).
5. `test_countries_analytics`: Asserts all 30 countries are aggregated and returned.
6. `test_country_detail`: Asserts single-country route returns 74 historical records for India.

---

### `tests/test_models.py` — Model Validation & Integrity Tests
Validates dataset integrity and machine learning consistency:
1. `test_feature_columns_loaded`: Verifies that all 10 expected features are present in the dataset.
2. `test_models_exist`: Verifies that all 3 classifiers and the forecaster bundle are loaded in memory.
3. `test_model_inference_consistency`: Iterates over Logistic Regression, Random Forest, and Gradient Boosting to verify that output probabilities are bounded in $[0.0, 1.0]$ and predictions are binary ($\{0, 1\}$).

---

## 8. DevOps & Containerization

### `Dockerfile` & `docker-compose.yml`
- **Dockerfile**:
  - Uses `python:3.11-slim` base image.
  - Installs dependencies from `requirements.txt`.
  - Exposes port `8008`.
  - Runs `python run_server.py`.
- **docker-compose.yml**:
  - Defines the `terrapulse` service with port mapping `8008:8008`.
  - Mounts local data and models directories for persistent execution.

---

## 9. Verification & Execution Summary

To run any part of the project:

```bash
# 1. Run the entire end-to-end data pipeline and train all models:
python main.py

# 2. Start the production server and dashboard:
python run_server.py

# 3. Run the automated test suite:
python -m pytest tests/ -v
```
All components are fully synchronized, self-contained, and production-ready.
