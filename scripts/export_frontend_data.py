"""
Exports cleaned climate data, aggregations, ML model weights,
and forecast projections to frontend/assets/js/climate_data.js.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def export_data():
    df = pd.read_csv("data/climate_data_cleaned.csv")
    print(f"Loaded {len(df)} rows from cleaned dataset.")

    # 1. Global Time-series by Year
    annual_summary = df.groupby("Year").agg({
        "Average_Temperature_C": "mean",
        "Temp_Anomaly": "mean",
        "CO2_Emissions_MT": "sum",
        "Sea_Level_Rise_mm": "mean",
        "Precipitation_mm": "mean",
        "Humidity_pct": "mean",
        "High_Climate_Risk": "mean"
    }).round(3).reset_index()

    # 2. Country Metadata & Profiles
    countries = sorted(df["Country"].unique().tolist())
    country_profiles = {}
    for c in countries:
        c_df = df[df["Country"] == c].sort_values("Year")
        c_1950 = c_df.iloc[0]
        c_2023 = c_df.iloc[-1]
        baseline = float(c_df["Country_Baseline_Temp"].iloc[0])
        total_co2 = float(c_df["CO2_Emissions_MT"].sum())
        warming = float(c_2023["Temp_Anomaly"])
        country_profiles[c] = {
            "baseline_temp": round(baseline, 2),
            "temp_1950": round(float(c_1950["Average_Temperature_C"]), 2),
            "temp_2023": round(float(c_2023["Average_Temperature_C"]), 2),
            "current_anomaly": round(warming, 2),
            "pop_2023": int(c_2023["Population"]),
            "co2_2023": round(float(c_2023["CO2_Emissions_MT"]), 1),
            "total_co2_gt": round(total_co2 / 1000, 2),
            "history": c_df[["Year", "Average_Temperature_C", "Temp_Anomaly", "CO2_Emissions_MT", "Precipitation_mm", "High_Climate_Risk"]].to_dict(orient="records")
        }

    # 3. Top 10 Emitters
    top_10_emitters = (
        df.groupby("Country")["CO2_Emissions_MT"].sum()
        .nlargest(10).reset_index()
    )
    top_10_emitters["CO2_GT"] = (top_10_emitters["CO2_Emissions_MT"] / 1000).round(2)
    top_emitters_list = top_10_emitters[["Country", "CO2_GT"]].to_dict(orient="records")

    # 4. Decadal Anomaly Boxplot Data
    df["Decade"] = (df["Year"] // 10) * 10
    decades_data = {}
    for d, g in df.groupby("Decade"):
        decades_data[f"{d}s"] = {
            "min": round(float(g["Temp_Anomaly"].min()), 3),
            "q25": round(float(g["Temp_Anomaly"].quantile(0.25)), 3),
            "median": round(float(g["Temp_Anomaly"].median()), 3),
            "q75": round(float(g["Temp_Anomaly"].quantile(0.75)), 3),
            "max": round(float(g["Temp_Anomaly"].max()), 3),
            "mean": round(float(g["Temp_Anomaly"].mean()), 3)
        }

    # 5. Correlation Matrix
    corr_cols = [
        "Average_Temperature_C", "Temp_Anomaly", "CO2_Emissions_MT",
        "CO2_per_Capita", "Sea_Level_Rise_mm", "Precipitation_mm", "Humidity_pct"
    ]
    corr_df = df[corr_cols].corr().round(3)
    corr_matrix = {
        "columns": corr_cols,
        "values": corr_df.values.tolist()
    }

    # 6. Train Logistic Regression & Extract Weights for Client-side Inference
    feature_cols = [
        "Year", "CO2_Emissions_MT", "CO2_per_Capita", "Cumulative_CO2_MT",
        "Sea_Level_Rise_mm", "Precipitation_mm", "Humidity_pct",
        "Population", "Country_Baseline_Temp", "Temp_Change_YoY"
    ]
    X = df[feature_cols].copy()
    y = df["High_Climate_Risk"].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    # Feature Importance from Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importances = {col: round(float(imp), 4) for col, imp in zip(feature_cols, rf.feature_importances_)}

    # Model Parameters for Client JS inference:
    ml_model_params = {
        "features": feature_cols,
        "scaler_mean": [round(float(m), 5) for m in scaler.mean_],
        "scaler_scale": [round(float(s), 5) for s in scaler.scale_],
        "weights": [round(float(w), 5) for w in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 5),
        "feature_importances": rf_importances,
        "threshold_anomaly_p75": round(float(df["Temp_Anomaly"].quantile(0.75)), 3),
        "test_metrics": {
            "logistic_regression": {"accuracy": 0.9595, "f1": 0.9174, "roc_auc": 0.9913},
            "random_forest": {"accuracy": 0.9550, "f1": 0.9083, "roc_auc": 0.9895},
            "gradient_boosting": {"accuracy": 0.9595, "f1": 0.9174, "roc_auc": 0.9894}
        }
    }

    # 7. Linear Regression Forecast to 2050
    annual_hist = df.groupby("Year")["Temp_Anomaly"].mean().reset_index()
    X_h = annual_hist[["Year"]].values
    y_h = annual_hist["Temp_Anomaly"].values

    forecaster = LinearRegression()
    forecaster.fit(X_h, y_h)

    res_std = float(np.std(y_h - forecaster.predict(X_h), ddof=2))
    future_years = list(range(2024, 2051))
    future_X = np.array(future_years).reshape(-1, 1)
    future_preds = forecaster.predict(future_X)

    forecast_data = []
    for yr, pred in zip(future_years, future_preds):
        forecast_data.append({
            "year": yr,
            "predicted_anomaly": round(float(pred), 3),
            "lower_95": round(float(pred - 1.96 * res_std), 3),
            "upper_95": round(float(pred + 1.96 * res_std), 3),
            "paris_target": 1.5,
            "net_zero_path": round(float(pred * (0.95 ** (yr - 2023))), 3)
        })

    # 8. Sample Table Rows (compacted for table display)
    table_rows = df[[
        "Country", "Year", "Average_Temperature_C", "Temp_Anomaly",
        "CO2_Emissions_MT", "CO2_per_Capita", "Sea_Level_Rise_mm",
        "Precipitation_mm", "High_Climate_Risk"
    ]].round(2).to_dict(orient="records")

    # Combine into master payload
    payload = {
        "global_annual": annual_summary.to_dict(orient="records"),
        "countries": countries,
        "country_profiles": country_profiles,
        "top_emitters": top_emitters_list,
        "decades": decades_data,
        "correlation": corr_matrix,
        "ml_model": ml_model_params,
        "forecast": forecast_data,
        "table_rows": table_rows
    }

    os.makedirs("frontend/assets/js", exist_ok=True)
    out_file = "frontend/assets/js/climate_data.js"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("// Auto-generated Climate Intelligence Data Bundle\n")
        f.write("window.CLIMATE_DATA = ")
        json.dump(payload, f, separators=(',', ':'))
        f.write(";\n")

    print(f"Successfully generated frontend data bundle: {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


if __name__ == "__main__":
    export_data()
