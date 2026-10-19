"""
Forecasting Module for Climate Change Analysis.
Fits a Linear Regression model on historical temperature anomaly trends (1950-2023)
and projects warming trajectories up to 2050 with confidence intervals.
"""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def save_forecaster(model, slope, residual_std, intercept=None, output_dir: str = "outputs/models"):
    """Serializes forecaster model and statistical parameters into a bundle."""
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        "model": model,
        "slope": float(slope),
        "residual_std": float(residual_std),
        "intercept": float(intercept) if intercept is not None else float(model.intercept_)
    }
    fpath = os.path.join(output_dir, "temperature_forecaster.joblib")
    joblib.dump(bundle, fpath)
    print(f"Saved temperature forecaster bundle to {fpath}")


def forecast_temperature_to_2050(df: pd.DataFrame, end_forecast_year: int = 2050, output_dir: str = "outputs/figures", models_dir: str = "outputs/models"):
    """
    Computes global annual temperature anomaly, fits linear regression,
    projects warming to 2050 with 95% confidence intervals, and serializes the model.
    """
    # Global annual mean anomaly
    annual_history = df.groupby("Year")["Temp_Anomaly"].mean().reset_index()

    X_hist = annual_history[["Year"]].values
    y_hist = annual_history["Temp_Anomaly"].values

    # Fit Linear Regression model
    model = LinearRegression()
    model.fit(X_hist, y_hist)

    slope = model.coef_[0]
    intercept = model.intercept_

    # Residual standard deviation for confidence bounds
    y_hist_pred = model.predict(X_hist)
    residuals = y_hist - y_hist_pred
    residual_std = np.std(residuals, ddof=2)

    # Future projection range
    future_years = np.arange(2024, end_forecast_year + 1).reshape(-1, 1)
    y_future_pred = model.predict(future_years)

    # 95% approximate prediction interval (1.96 * std)
    lower_bound = y_future_pred - 1.96 * residual_std
    upper_bound = y_future_pred + 1.96 * residual_std

    forecast_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted_Anomaly": np.round(y_future_pred, 3),
        "Lower_Bound_95": np.round(lower_bound, 3),
        "Upper_Bound_95": np.round(upper_bound, 3)
    })

    # Plot forecast
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    
    # Historical curve
    ax.plot(annual_history["Year"], annual_history["Temp_Anomaly"], color="#2b5c8f", marker="o", markersize=3.5,
            linewidth=2, label="Historical Global Anomaly (1950–2023)")
    
    # Historical fitted trend
    ax.plot(annual_history["Year"], y_hist_pred, color="#6baed6", linestyle=":", linewidth=1.8, label="Historical Linear Fit")
    
    # Forecast curve
    ax.plot(forecast_df["Year"], forecast_df["Predicted_Anomaly"], color="#cb181d", linestyle="--", linewidth=2.5,
            label=f"Projected Linear Forecast (2024–2050, +{slope*10:.3f}°C/decade)")
    
    # Confidence envelope
    ax.fill_between(forecast_df["Year"], forecast_df["Lower_Bound_95"], forecast_df["Upper_Bound_95"],
                    color="#fb6a4a", alpha=0.25, label="95% Confidence Interval")

    # Critical thresholds (e.g. 1.5°C threshold)
    ax.axhline(1.5, color="#800026", linestyle="-.", linewidth=1.2, alpha=0.8, label="1.5°C Warming Threshold")
    
    # Annotate 2050 projection
    val_2050 = forecast_df["Predicted_Anomaly"].iloc[-1]
    ax.scatter([2050], [val_2050], color="#cb181d", s=70, zorder=6)
    ax.annotate(f"2050 Projection: +{val_2050:.2f}°C",
                xy=(2050, val_2050), xytext=(2035, val_2050 + 0.35),
                arrowprops=dict(facecolor="#cb181d", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", color="#cb181d", fontsize=9.5)

    ax.set_title("Global Temperature Anomaly Projection to 2050 (Linear Regression)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Temperature Anomaly (°C)", fontsize=10, labelpad=8)
    ax.set_xlim(1948, 2053)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "12_temperature_forecast_2050.png"))

    if models_dir:
        save_forecaster(model, slope, residual_std, intercept, output_dir=models_dir)

    return forecast_df, fig, model


if __name__ == "__main__":
    df = pd.read_csv("data/climate_data_cleaned.csv")
    forecast_df, fig, model = forecast_temperature_to_2050(df)
    print("\n--- Temperature Anomaly Projections to 2050 (Sample) ---")
    print(forecast_df.head(10).to_string(index=False))
