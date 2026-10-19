"""
Main Execution Script for Climate Change Data Analysis and Prediction.
Runs the entire end-to-end pipeline:
1. Synthetic Data Generation (2,220 rows across 30 countries from 1950 to 2023)
2. Data Cleaning with Country Median Imputation & Feature Engineering
3. Exploratory Data Analysis (8 publication-grade charts saved to outputs/figures/)
4. Machine Learning Classification (Logistic Regression, Random Forest, Gradient Boosting)
5. Long-term Temperature Forecasting to 2050 using Linear Regression
"""

import os
import sys

from src.data_generation import generate_climate_dataset, save_dataset
from src.data_cleaning import clean_climate_data, engineer_features
from src.eda import run_all_eda
from src.modeling import (
    prepare_data, train_and_evaluate_models,
    plot_model_evaluations, plot_feature_importance, save_models
)
from src.forecasting import forecast_temperature_to_2050
from scripts.build_notebook import build_notebook
from scripts.export_frontend_data import export_data


def run_pipeline():
    print("=" * 70)
    print("CLIMATE CHANGE DATA ANALYSIS AND PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Data Generation
    print("\n[Step 1/6] Generating Synthetic Climate Dataset (30 countries, 1950-2023)...")
    df_raw = generate_climate_dataset()
    save_dataset(df_raw, "data/climate_data_raw.csv")
    print(f"-> Generated {len(df_raw)} records. Missing values simulated.")

    # 2. Data Cleaning & Feature Engineering
    print("\n[Step 2/6] Cleaning Data (Country-specific Median Imputation) & Feature Engineering...")
    df_clean = clean_climate_data(df_raw)
    df_featured = engineer_features(df_clean)
    os.makedirs("data", exist_ok=True)
    df_featured.to_csv("data/climate_data_cleaned.csv", index=False)
    print(f"-> Cleaned data saved to 'data/climate_data_cleaned.csv' (Missing values: {df_featured.isnull().sum().sum()}).")

    # 3. Exploratory Data Analysis
    print("\n[Step 3/6] Generating 8 Visually Appealing EDA Charts...")
    os.makedirs("outputs/figures", exist_ok=True)
    run_all_eda(df_featured, output_dir="outputs/figures")

    # 4. Machine Learning Classification & Serialization
    print("\n[Step 4/6] Training & Evaluating ML Models (Logistic Regression, Random Forest, Gradient Boosting)...")
    X_train, X_test, y_train, y_test = prepare_data(df_featured)
    metrics_df, trained_models, predictions, probabilities = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    print("\n--- Model Performance Summary ---")
    print(metrics_df.to_string(index=False))

    plot_model_evaluations(y_test, probabilities, predictions, output_dir="outputs/figures")
    plot_feature_importance(trained_models, output_dir="outputs/figures")
    save_models(trained_models, metrics_df, output_dir="outputs/models", training_records=len(df_featured))
    print("-> Evaluation charts and serialized classification models saved to 'outputs/models/'.")

    # 5. Long-term Climate Forecasting to 2050
    print("\n[Step 5/6] Forecasting Global Temperature Anomaly Trajectory to 2050...")
    forecast_df, fig, model = forecast_temperature_to_2050(
        df_featured, end_forecast_year=2050, output_dir="outputs/figures", models_dir="outputs/models"
    )
    print("\n--- Projected Warming (Sample Years) ---")
    sample_years = forecast_df[forecast_df["Year"].isin([2025, 2030, 2035, 2040, 2045, 2050])]
    print(sample_years.to_string(index=False))

    # 6. Artifact Synchronization (Frontend Data & Notebook)
    print("\n[Step 6/6] Synchronizing Frontend Data Bundle & Building Jupyter Notebook...")
    export_data()
    build_notebook()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Artifacts generated & synchronized in:")
    print("  - Data files: data/climate_data_raw.csv, data/climate_data_cleaned.csv")
    print("  - Visualizations: outputs/figures/ (12 high-resolution charts)")
    print("  - Models: outputs/models/ (4 joblib models + model_metadata.json)")
    print("  - Frontend Data: frontend/assets/js/climate_data.js")
    print("  - Notebook: notebooks/02_climate_analysis_and_prediction.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
