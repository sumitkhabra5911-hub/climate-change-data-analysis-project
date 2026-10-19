"""
Service layer for ML inference, forecasting, and data querying.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.config import settings
from src.modeling import FEATURE_COLS


class ModelService:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.forecaster_bundle: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.df: Optional[pd.DataFrame] = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized models, metadata, and dataset."""
        # Load dataset
        if os.path.exists(settings.DATA_PATH):
            self.df = pd.read_csv(settings.DATA_PATH)
        else:
            self.df = pd.DataFrame()

        # Load metadata
        meta_path = os.path.join(settings.MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Load classification models
        model_files = {
            "Logistic Regression": "logistic_regression.joblib",
            "Random Forest": "random_forest.joblib",
            "Gradient Boosting": "gradient_boosting.joblib"
        }
        for name, fname in model_files.items():
            fpath = os.path.join(settings.MODELS_DIR, fname)
            if os.path.exists(fpath):
                self.models[name] = joblib.load(fpath)

        # Load forecaster
        fpath = os.path.join(settings.MODELS_DIR, "temperature_forecaster.joblib")
        if os.path.exists(fpath):
            self.forecaster_bundle = joblib.load(fpath)

    def predict_risk(self, features_dict: Dict[str, Any], model_name: str = "Logistic Regression") -> Dict[str, Any]:
        """Performs real-time ML risk classification and feature attribution."""
        if model_name not in self.models:
            model_name = "Logistic Regression"
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' is not loaded.")

        X = pd.DataFrame([[features_dict[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)

        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        # Feature impact / attribution
        contributions = []
        if model_name == "Logistic Regression":
            scaler = model.named_steps["scaler"]
            clf = model.named_steps["clf"]
            x_scaled = scaler.transform(X)[0]
            coeffs = clf.coef_[0]
            for col, val, c in zip(FEATURE_COLS, x_scaled, coeffs):
                impact = float(val * c)
                contributions.append({
                    "feature": col,
                    "impact_score": round(abs(impact), 4),
                    "direction": "increases_risk" if impact > 0 else "decreases_risk"
                })
        else:
            importances = model.feature_importances_
            for col, imp in zip(FEATURE_COLS, importances):
                contributions.append({
                    "feature": col,
                    "impact_score": round(float(imp), 4),
                    "direction": "increases_risk"
                })

        contributions.sort(key=lambda x: x["impact_score"], reverse=True)

        return {
            "model_used": model_name,
            "risk_prediction": pred,
            "risk_label": "High Climate Risk Acceleration" if pred == 1 else "Normal / Controlled Risk",
            "risk_probability": round(prob, 4),
            "confidence_percentage": round(prob * 100 if pred == 1 else (1 - prob) * 100, 2),
            "top_risk_factors": contributions[:5]
        }

    def predict_forecast(self, target_year: int, scenario: str = "business_as_usual") -> Dict[str, Any]:
        """Projects temperature anomaly & sea level rise based on policy scenarios."""
        bundle = self.forecaster_bundle
        if not bundle:
            raise ValueError("Forecaster model is not loaded.")

        model = bundle["model"]
        slope = bundle["slope"]
        residual_std = bundle["residual_std"]

        # Base BAU prediction
        bau_pred = float(model.predict([[target_year]])[0])

        # Adjust for mitigation scenarios
        years_ahead = max(0, target_year - 2023)
        if scenario == "paris_agreement":
            # Decelerates warming towards +1.5C stabilization
            adjusted_pred = min(bau_pred, 1.45 + 0.05 * (1 - np.exp(-years_ahead / 15.0)))
        elif scenario == "net_zero":
            # Net Zero by 2050 stabilizes around +1.35C then plateaus
            adjusted_pred = min(bau_pred, 1.30 + 0.15 * (1 - np.exp(-years_ahead / 12.0)))
        else:
            scenario = "business_as_usual"
            adjusted_pred = bau_pred

        lower = round(adjusted_pred - 1.96 * residual_std, 3)
        upper = round(adjusted_pred + 1.96 * residual_std, 3)
        projected = round(adjusted_pred, 3)

        # Empirical sea level rise proxy (approx 3.4 mm/yr accelerating with warming)
        sea_level = round(105.0 + years_ahead * (3.8 + (projected - 1.2) * 1.5), 1)

        return {
            "target_year": target_year,
            "scenario": scenario,
            "projected_anomaly_celsius": projected,
            "lower_bound_95": lower,
            "upper_bound_95": upper,
            "estimated_sea_level_rise_mm": sea_level,
            "warming_rate_celsius_per_decade": round(slope * 10, 4)
        }

    def get_countries_summary(self) -> List[Dict[str, Any]]:
        """Aggregates historical metrics across all countries."""
        if self.df is None or self.df.empty:
            return []
        
        grouped = []
        for country, sub in self.df.groupby("Country"):
            latest = sub.sort_values("Year").iloc[-1]
            grouped.append({
                "country": country,
                "mean_temp_anomaly": round(float(sub["Temp_Anomaly"].mean()), 3),
                "max_temp_anomaly": round(float(sub["Temp_Anomaly"].max()), 3),
                "latest_temp_anomaly": round(float(latest["Temp_Anomaly"]), 3),
                "latest_co2_mt": round(float(latest["CO2_Emissions_MT"]), 2),
                "cumulative_co2_mt": round(float(latest["Cumulative_CO2_MT"]), 2),
                "high_risk_years_count": int(sub["High_Climate_Risk"].sum()),
                "records_count": len(sub)
            })
        return sorted(grouped, key=lambda x: x["latest_co2_mt"], reverse=True)

    def get_country_detail(self, country_name: str) -> Dict[str, Any]:
        """Returns time series records and stats for a single country."""
        if self.df is None or self.df.empty:
            raise ValueError("Dataset not loaded.")

        sub = self.df[self.df["Country"].str.lower() == country_name.lower()].sort_values("Year")
        if sub.empty:
            raise KeyError(f"Country '{country_name}' not found.")

        latest = sub.iloc[-1]
        summary = {
            "country": sub["Country"].iloc[0],
            "mean_temp_anomaly": round(float(sub["Temp_Anomaly"].mean()), 3),
            "max_temp_anomaly": round(float(sub["Temp_Anomaly"].max()), 3),
            "latest_temp_anomaly": round(float(latest["Temp_Anomaly"]), 3),
            "latest_co2_mt": round(float(latest["CO2_Emissions_MT"]), 2),
            "cumulative_co2_mt": round(float(latest["Cumulative_CO2_MT"]), 2),
            "high_risk_years_count": int(sub["High_Climate_Risk"].sum()),
            "records_count": len(sub)
        }
        return {
            "country": summary["country"],
            "records": sub.to_dict(orient="records"),
            "statistics": summary
        }

model_service = ModelService()
