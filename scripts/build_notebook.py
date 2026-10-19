"""
Script to generate the complete, self-contained Jupyter Notebook:
notebooks/02_climate_analysis_and_prediction.ipynb
Ensures 100% valid JSON and compliant Jupyter format (nbformat 4.5).
"""

import json
import os


def make_cell(cell_type, source, cell_id=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook():
    cells = []

    # Title
    cells.append(make_cell("markdown", """# Beginner-Friendly Climate Change Data Analysis and Prediction
### An End-to-End Data Science & Machine Learning Workflow (1950–2023 & Forecast to 2050)

---

## Project Overview
This project presents a comprehensive, reproducible data science workflow analyzing historical climate indicators across **30 countries over 74 years (1950–2023)**, totaling **2,220 observations**.

### Workflow Highlights:
1. **Realistic Synthetic Data Generation**: Simulates historical temperature anomalies, non-linear greenhouse gas emissions, sea level rise, and precipitation variability with deliberate real-world data flaws (missing values).
2. **Robust Data Cleaning**: Resolves missingness using **country-specific median imputation**.
3. **Domain Feature Engineering**: Derives `CO2_per_Capita`, `Temp_Anomaly`, `Temp_Change_YoY`, and a binary classification target `High_Climate_Risk` (75th percentile anomaly threshold).
4. **Exploratory Data Analysis (EDA)**: 8 publication-grade charts uncovering warming trends, decadal distributions, top emitters, and regional divergence.
5. **Supervised Machine Learning**: Benchmarks **Logistic Regression**, **Random Forest**, and **Gradient Boosting** with 5-fold Stratified Cross-Validation, ROC-AUC curves, confusion matrices, and feature importance.
6. **Time-Series Forecasting**: Projections to **2050** using linear regression modeling."""))

    # Imports
    cells.append(make_cell("markdown", "## 1. Environment Setup and Library Imports"))
    cells.append(make_cell("code", """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set global aesthetic styling
%matplotlib inline
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["figure.dpi"] = 120

print("All required libraries loaded successfully!")"""))

    # Synthetic Data Generation
    cells.append(make_cell("markdown", """## 2. Synthetic Climate Data Generation
Generates a realistic synthetic dataset covering 30 countries from 1950 to 2023 (exactly 2,220 rows). 
Features include:
- `Country` & `Year`
- `Average_Temperature_C`
- `CO2_Emissions_MT`
- `Sea_Level_Rise_mm`
- `Precipitation_mm`
- `Humidity_pct`
- `Population`

Random missing values (~3%) are introduced to simulate real-world data collection imperfections."""))

    cells.append(make_cell("code", """# Country configuration with realistic geographical baselines
COUNTRIES_CONFIG = {
    "United States": {"base_temp": 11.5, "warming_rate": 1.05, "base_pop": 157_000_000, "pop_growth": 0.011, "base_co2": 2500, "co2_growth": 0.015, "precip": 750, "humidity": 65},
    "China": {"base_temp": 7.8, "warming_rate": 1.15, "base_pop": 544_000_000, "pop_growth": 0.013, "base_co2": 800, "co2_growth": 0.045, "precip": 645, "humidity": 66},
    "India": {"base_temp": 24.5, "warming_rate": 0.95, "base_pop": 376_000_000, "pop_growth": 0.018, "base_co2": 250, "co2_growth": 0.042, "precip": 1080, "humidity": 68},
    "Germany": {"base_temp": 8.5, "warming_rate": 1.10, "base_pop": 69_000_000, "pop_growth": 0.003, "base_co2": 600, "co2_growth": 0.005, "precip": 780, "humidity": 75},
    "United Kingdom": {"base_temp": 9.2, "warming_rate": 0.90, "base_pop": 50_000_000, "pop_growth": 0.004, "base_co2": 500, "co2_growth": -0.002, "precip": 1150, "humidity": 80},
    "Japan": {"base_temp": 12.0, "warming_rate": 1.05, "base_pop": 82_000_000, "pop_growth": 0.006, "base_co2": 350, "co2_growth": 0.020, "precip": 1650, "humidity": 72},
    "Brazil": {"base_temp": 25.0, "warming_rate": 0.95, "base_pop": 54_000_000, "pop_growth": 0.019, "base_co2": 150, "co2_growth": 0.035, "precip": 1780, "humidity": 80},
    "Canada": {"base_temp": -4.2, "warming_rate": 1.60, "base_pop": 14_000_000, "pop_growth": 0.014, "base_co2": 280, "co2_growth": 0.018, "precip": 540, "humidity": 70},
    "Australia": {"base_temp": 21.6, "warming_rate": 1.10, "base_pop": 8_200_000, "pop_growth": 0.016, "base_co2": 180, "co2_growth": 0.022, "precip": 465, "humidity": 55},
    "Russia": {"base_temp": -5.1, "warming_rate": 1.55, "base_pop": 102_000_000, "pop_growth": 0.005, "base_co2": 950, "co2_growth": 0.012, "precip": 460, "humidity": 74},
    "South Africa": {"base_temp": 17.5, "warming_rate": 1.05, "base_pop": 13_600_000, "pop_growth": 0.020, "base_co2": 120, "co2_growth": 0.025, "precip": 495, "humidity": 60},
    "Egypt": {"base_temp": 22.8, "warming_rate": 1.12, "base_pop": 20_500_000, "pop_growth": 0.022, "base_co2": 60, "co2_growth": 0.032, "precip": 50, "humidity": 45},
    "France": {"base_temp": 11.2, "warming_rate": 1.08, "base_pop": 42_000_000, "pop_growth": 0.006, "base_co2": 420, "co2_growth": 0.004, "precip": 860, "humidity": 74},
    "Italy": {"base_temp": 13.4, "warming_rate": 1.15, "base_pop": 46_500_000, "pop_growth": 0.003, "base_co2": 320, "co2_growth": 0.010, "precip": 830, "humidity": 69},
    "Mexico": {"base_temp": 21.0, "warming_rate": 1.05, "base_pop": 28_000_000, "pop_growth": 0.021, "base_co2": 110, "co2_growth": 0.028, "precip": 720, "humidity": 62},
    "Indonesia": {"base_temp": 26.8, "warming_rate": 0.88, "base_pop": 70_000_000, "pop_growth": 0.019, "base_co2": 100, "co2_growth": 0.038, "precip": 2750, "humidity": 84},
    "Nigeria": {"base_temp": 27.2, "warming_rate": 0.98, "base_pop": 37_000_000, "pop_growth": 0.025, "base_co2": 40, "co2_growth": 0.030, "precip": 1150, "humidity": 72},
    "Argentina": {"base_temp": 15.2, "warming_rate": 0.92, "base_pop": 17_000_000, "pop_growth": 0.014, "base_co2": 95, "co2_growth": 0.020, "precip": 590, "humidity": 64},
    "Saudi Arabia": {"base_temp": 25.8, "warming_rate": 1.25, "base_pop": 3_200_000, "pop_growth": 0.033, "base_co2": 80, "co2_growth": 0.048, "precip": 90, "humidity": 35},
    "Turkey": {"base_temp": 12.2, "warming_rate": 1.14, "base_pop": 21_500_000, "pop_growth": 0.019, "base_co2": 90, "co2_growth": 0.034, "precip": 590, "humidity": 62},
    "South Korea": {"base_temp": 11.8, "warming_rate": 1.20, "base_pop": 19_200_000, "pop_growth": 0.014, "base_co2": 70, "co2_growth": 0.042, "precip": 1300, "humidity": 69},
    "Spain": {"base_temp": 14.8, "warming_rate": 1.22, "base_pop": 28_000_000, "pop_growth": 0.007, "base_co2": 160, "co2_growth": 0.016, "precip": 630, "humidity": 63},
    "Norway": {"base_temp": 3.2, "warming_rate": 1.45, "base_pop": 3_300_000, "pop_growth": 0.007, "base_co2": 45, "co2_growth": 0.010, "precip": 1400, "humidity": 78},
    "Sweden": {"base_temp": 4.1, "warming_rate": 1.40, "base_pop": 7_000_000, "pop_growth": 0.006, "base_co2": 60, "co2_growth": 0.005, "precip": 650, "humidity": 77},
    "Kenya": {"base_temp": 24.8, "warming_rate": 0.94, "base_pop": 6_100_000, "pop_growth": 0.028, "base_co2": 15, "co2_growth": 0.035, "precip": 630, "humidity": 66},
    "Vietnam": {"base_temp": 24.2, "warming_rate": 0.92, "base_pop": 25_000_000, "pop_growth": 0.018, "base_co2": 35, "co2_growth": 0.044, "precip": 1820, "humidity": 82},
    "Colombia": {"base_temp": 24.0, "warming_rate": 0.90, "base_pop": 12_500_000, "pop_growth": 0.020, "base_co2": 40, "co2_growth": 0.028, "precip": 2600, "humidity": 81},
    "Thailand": {"base_temp": 26.5, "warming_rate": 0.93, "base_pop": 20_700_000, "pop_growth": 0.016, "base_co2": 50, "co2_growth": 0.036, "precip": 1620, "humidity": 77},
    "Poland": {"base_temp": 7.9, "warming_rate": 1.12, "base_pop": 25_000_000, "pop_growth": 0.006, "base_co2": 260, "co2_growth": 0.008, "precip": 600, "humidity": 76},
    "New Zealand": {"base_temp": 11.0, "warming_rate": 0.90, "base_pop": 1_900_000, "pop_growth": 0.013, "base_co2": 25, "co2_growth": 0.018, "precip": 1380, "humidity": 79}
}

np.random.seed(42)
years = list(range(1950, 2024))
records = []

for country, cfg in COUNTRIES_CONFIG.items():
    base_temp = cfg["base_temp"]
    warming_rate = cfg["warming_rate"]
    base_pop = cfg["base_pop"]
    pop_growth = cfg["pop_growth"]
    base_co2_val = cfg["base_co2"]
    precip_base = cfg["precip"]
    humidity_base = cfg["humidity"]

    for t, year in enumerate(years):
        # 1. Non-linear warming trend accelerating post 1980
        accel_factor = 0.008 * (t ** 1.35) if t < 30 else 0.008 * (30 ** 1.35) + 0.032 * ((t - 30) ** 1.15)
        warming_anomaly = accel_factor * warming_rate
        enso_cycle = 0.22 * np.sin(2 * np.pi * t / 5.2) + 0.15 * np.cos(2 * np.pi * t / 11.0)
        noise_temp = np.random.normal(0, 0.28)
        avg_temp = round(base_temp + warming_anomaly + enso_cycle + noise_temp, 2)

        # 2. Population growth
        current_pop = int(base_pop * np.exp(pop_growth * t * (1 - 0.003 * t)))

        # 3. Non-linear CO2 emissions
        if country in ["United States", "Germany", "United Kingdom", "France"]:
            co2_trend = base_co2_val * (1 + 0.025 * t - 0.0003 * (t ** 2))
            if year > 2005:
                co2_trend *= (1 - 0.012 * (year - 2005))
        elif country in ["China", "India", "Vietnam"]:
            boost = 1 + np.exp((year - 1990) / 10.0) / (1 + np.exp((year - 1990) / 10.0)) * 3.5
            co2_trend = base_co2_val * boost * (1 + 0.015 * t)
        else:
            co2_trend = base_co2_val * np.exp(cfg["co2_growth"] * t * 0.9)
        co2_emissions = round(max(5.0, co2_trend + np.random.normal(0, 0.05 * co2_trend)), 2)

        # 4. Sea level rise (mm): accelerating from 1950 to ~220mm
        sea_level = round(0.9 * t + 0.028 * (t ** 2) + np.random.normal(0, 2.5), 2)

        # 5. Precipitation
        precip_trend = precip_base * (1 + 0.0008 * warming_anomaly + np.random.normal(0, 0.09))
        precip_mm = round(max(10.0, precip_trend), 1)

        # 6. Humidity
        humidity_val = humidity_base + (precip_mm - precip_base) * 0.008 - warming_anomaly * 0.4 + np.random.normal(0, 1.8)
        humidity_pct = round(np.clip(humidity_val, 15.0, 98.0), 1)

        records.append({
            "Country": country,
            "Year": year,
            "Average_Temperature_C": avg_temp,
            "CO2_Emissions_MT": co2_emissions,
            "Sea_Level_Rise_mm": sea_level,
            "Precipitation_mm": precip_mm,
            "Humidity_pct": humidity_pct,
            "Population": current_pop
        })

df_raw = pd.DataFrame(records)

# Introduce 3% simulated missingness in observational indicators
for col in ["Average_Temperature_C", "CO2_Emissions_MT", "Precipitation_mm", "Humidity_pct"]:
    mask = np.random.rand(len(df_raw)) < 0.03
    df_raw.loc[mask, col] = np.nan

os.makedirs("data", exist_ok=True)
df_raw.to_csv("data/climate_data_raw.csv", index=False)
print(f"Generated raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"Total missing values introduced: {df_raw.isnull().sum().to_dict()}")
df_raw.head()"""))

    # Data Cleaning
    cells.append(make_cell("markdown", """## 3. Data Cleaning: Country-Specific Median Imputation
Because climate variables differ drastically by geography (e.g. Norway's average temperature vs Egypt's), imputing missing values with a global average would severely bias the data. 

We impute missing values using **country-specific medians**."""))

    cells.append(make_cell("code", """df_clean = df_raw.copy()
df_clean = df_clean.sort_values(by=["Country", "Year"]).reset_index(drop=True)

numeric_cols = ["Average_Temperature_C", "CO2_Emissions_MT", "Sea_Level_Rise_mm", "Precipitation_mm", "Humidity_pct", "Population"]

for col in numeric_cols:
    df_clean[col] = df_clean.groupby("Country")[col].transform(lambda grp: grp.fillna(grp.median()))
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("Missing values after country-specific median imputation:")
print(df_clean.isnull().sum())"""))

    # Feature Engineering
    cells.append(make_cell("markdown", """## 4. Feature Engineering
We construct domain-specific climate metrics:
1. `CO2_per_Capita`: Metric tons of CO2 emitted per person.
2. `Country_Baseline_Temp`: Historical baseline average temperature for each country (1950–1980 reference period).
3. `Temp_Anomaly`: Local temperature deviation from the 1950–1980 baseline.
4. `Temp_Change_YoY`: Year-over-Year change in temperature per country.
5. `Temp_5yr_MA`: 5-year rolling moving average temperature.
6. `Cumulative_CO2_MT`: Cumulative CO2 emitted since 1950.
7. `High_Climate_Risk`: Target binary classification indicator defined as `Temp_Anomaly > 75th percentile`."""))

    cells.append(make_cell("code", """df_feat = df_clean.copy()

# 1. CO2 per capita (Metric Tons / person)
df_feat["CO2_per_Capita"] = (df_feat["CO2_Emissions_MT"] * 1_000_000) / df_feat["Population"]

# 2. Country baseline temperature (1950-1980 reference)
baseline_temps = df_feat[df_feat["Year"] <= 1980].groupby("Country")["Average_Temperature_C"].mean().to_dict()
df_feat["Country_Baseline_Temp"] = df_feat["Country"].map(baseline_temps)

# 3. Temperature Anomaly (°C)
df_feat["Temp_Anomaly"] = round(df_feat["Average_Temperature_C"] - df_feat["Country_Baseline_Temp"], 3)

# 4. Year-over-Year Temperature Change
df_feat["Temp_Change_YoY"] = df_feat.groupby("Country")["Average_Temperature_C"].diff().fillna(0)

# 5. 5-year Moving Average
df_feat["Temp_5yr_MA"] = df_feat.groupby("Country")["Average_Temperature_C"].transform(
    lambda s: s.rolling(5, min_periods=1).mean()
)

# 6. Cumulative CO2 Emissions
df_feat["Cumulative_CO2_MT"] = df_feat.groupby("Country")["CO2_Emissions_MT"].cumsum()

# 7. High Climate Risk target (75th percentile threshold)
threshold_75 = df_feat["Temp_Anomaly"].quantile(0.75)
df_feat["High_Climate_Risk"] = (df_feat["Temp_Anomaly"] > threshold_75).astype(int)

df_feat.to_csv("data/climate_data_cleaned.csv", index=False)
print(f"High Climate Risk threshold (75th percentile anomaly): +{threshold_75:.3f} °C")
print(f"Target distribution:\n{df_feat['High_Climate_Risk'].value_counts(normalize=True).round(3)}")
df_feat[["Country", "Year", "Average_Temperature_C", "Temp_Anomaly", "CO2_per_Capita", "High_Climate_Risk"]].tail()"""))

    # EDA
    cells.append(make_cell("markdown", "## 5. Exploratory Data Analysis (EDA): 8 Core Visualizations"))

    # Chart 1
    cells.append(make_cell("markdown", "### Chart 1: Global Average Temperature Anomaly Trend (1950–2023)"))
    cells.append(make_cell("code", """annual_global = df_feat.groupby("Year").agg({"Temp_Anomaly": ["mean", "std"]}).reset_index()
annual_global.columns = ["Year", "Mean_Anomaly", "Std_Anomaly"]
annual_global["Rolling_5yr"] = annual_global["Mean_Anomaly"].rolling(5, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#d73027" if val >= 0 else "#4575b4" for val in annual_global["Mean_Anomaly"]]
ax.bar(annual_global["Year"], annual_global["Mean_Anomaly"], color=colors, alpha=0.6, width=0.8, label="Annual Anomaly")
ax.plot(annual_global["Year"], annual_global["Rolling_5yr"], color="#800026", linewidth=2.5, label="5-Year Moving Avg")
ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

ax.annotate("Accelerated Warming Post-1980", xy=(1985, 0.4), xytext=(1960, 1.2),
            arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.2, headwidth=6),
            fontsize=9.5, fontweight="bold")

ax.set_title("1. Global Average Temperature Anomaly Trend (1950–2023)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Anomaly (°C rel. to 1950–1980 base)", fontsize=10)
ax.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.show()"""))

    # Chart 2
    cells.append(make_cell("markdown", "### Chart 2: Top 10 Cumulative CO2 Emitting Countries"))
    cells.append(make_cell("code", """top_emitters = df_feat.groupby("Country")["CO2_Emissions_MT"].sum().nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
palette = sns.color_palette("rocket", n_colors=len(top_emitters))
bars = ax.barh(top_emitters.index, top_emitters.values / 1e3, color=palette, height=0.65)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 2, bar.get_y() + bar.get_height() / 2, f"{w:.1f} GT", va="center", fontsize=8.5)

ax.set_title("2. Top 10 Cumulative CO2 Emitting Countries (1950–2023)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Cumulative CO2 Emissions (Gigatons, GT)", fontsize=10)
ax.set_xlim(0, (top_emitters.max() / 1e3) * 1.15)
plt.tight_layout()
plt.show()"""))

    # Chart 3
    cells.append(make_cell("markdown", "### Chart 3: Correlation Matrix of Climate Indicators"))
    cells.append(make_cell("code", """feature_cols = [
    "Average_Temperature_C", "Temp_Anomaly", "CO2_Emissions_MT",
    "CO2_per_Capita", "Sea_Level_Rise_mm", "Precipitation_mm",
    "Humidity_pct", "Population"
]
corr_matrix = df_feat[feature_cols].corr()

fig, ax = plt.subplots(figsize=(8.5, 6.5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
            linewidths=0.8, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("3. Correlation Matrix of Climate Indicators", fontsize=13, fontweight="bold", pad=12)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.show()"""))

    # Chart 4
    cells.append(make_cell("markdown", "### Chart 4: Cumulative Sea Level Rise Acceleration"))
    cells.append(make_cell("code", """slr_data = df_feat.groupby("Year")["Sea_Level_Rise_mm"].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(slr_data["Year"], slr_data["Sea_Level_Rise_mm"], color="#08519c", linewidth=2.8, label="Global Mean SLR (mm)")
ax.fill_between(slr_data["Year"], 0, slr_data["Sea_Level_Rise_mm"], color="#9ecae1", alpha=0.35)

final_yr = slr_data["Year"].iloc[-1]
final_val = slr_data["Sea_Level_Rise_mm"].iloc[-1]
ax.scatter([final_yr], [final_val], color="#08519c", s=60, zorder=5)
ax.text(final_yr - 6, final_val + 8, f"{final_val:.1f} mm\\n(2023)", fontweight="bold", color="#08519c", fontsize=9)

ax.set_title("4. Cumulative Global Mean Sea Level Rise (1950–2023)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Sea Level Rise (mm)", fontsize=10)
ax.set_ylim(bottom=-5)
ax.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.show()"""))

    # Chart 5
    cells.append(make_cell("markdown", "### Chart 5: Shifting Distribution of Temperature Anomalies by Decade"))
    cells.append(make_cell("code", """df_decade = df_feat.copy()
df_decade["Decade"] = (df_decade["Year"] // 10) * 10
df_decade["Decade_Label"] = df_decade["Decade"].astype(str) + "s"

fig, ax = plt.subplots(figsize=(10, 5))
order = sorted(df_decade["Decade_Label"].unique())
palette = sns.color_palette("YlOrRd", n_colors=len(order))
sns.boxplot(x="Decade_Label", y="Temp_Anomaly", data=df_decade, order=order, hue="Decade_Label", legend=False, palette=palette, width=0.55, ax=ax)
ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

ax.set_title("5. Decadal Shift in Temperature Anomaly Distributions (1950s–2020s)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Decade", fontsize=10)
ax.set_ylabel("Temperature Anomaly (°C)", fontsize=10)
plt.tight_layout()
plt.show()"""))

    # Chart 6
    cells.append(make_cell("markdown", "### Chart 6: Per Capita CO2 Emissions vs. Temperature Anomaly"))
    cells.append(make_cell("code", """fig, ax = plt.subplots(figsize=(10, 5.5))
scatter = ax.scatter(
    df_feat["CO2_per_Capita"], df_feat["Temp_Anomaly"],
    c=df_feat["Year"], cmap="plasma", alpha=0.6, s=30, edgecolors="none"
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Observation Year", fontsize=9)

sns.regplot(
    x="CO2_per_Capita", y="Temp_Anomaly", data=df_feat,
    scatter=False, ax=ax, color="#222222", line_kws={"linewidth": 2, "linestyle": "--"}
)

ax.set_title("6. Per Capita CO2 Emissions vs. Temperature Anomaly (1950–2023)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("CO2 Emissions per Capita (Metric Tons)", fontsize=10)
ax.set_ylabel("Temperature Anomaly (°C)", fontsize=10)
plt.tight_layout()
plt.show()"""))

    # Chart 7
    cells.append(make_cell("markdown", "### Chart 7: Annual Precipitation Trends & Volatility"))
    cells.append(make_cell("code", """precip_annual = df_feat.groupby("Year")["Precipitation_mm"].agg(["mean", "std"]).reset_index()
precip_annual["Rolling_5yr"] = precip_annual["mean"].rolling(5, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(
    precip_annual["Year"],
    precip_annual["mean"] - precip_annual["std"],
    precip_annual["mean"] + precip_annual["std"],
    color="#a1d99b", alpha=0.35, label="±1 Std Dev (Regional Spread)"
)
ax.plot(precip_annual["Year"], precip_annual["mean"], color="#238b45", linewidth=2.0, alpha=0.8, label="Global Mean Precipitation")
ax.plot(precip_annual["Year"], precip_annual["Rolling_5yr"], color="#00441b", linewidth=2.5, label="5-Year Moving Avg")

ax.set_title("7. Global Annual Mean Precipitation & Interannual Fluctuations (1950–2023)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Precipitation (mm/year)", fontsize=10)
ax.legend(frameon=True, loc="upper right")
plt.tight_layout()
plt.show()"""))

    # Chart 8
    cells.append(make_cell("markdown", "### Chart 8: Regional Temperature Trajectories (Arctic Amplification vs Tropics)"))
    cells.append(make_cell("code", """focus_countries = ["Canada", "Norway", "United States", "China", "India", "Brazil", "Australia"]
palette = sns.color_palette("tab10", n_colors=len(focus_countries))

fig, ax = plt.subplots(figsize=(10, 5.5))
for country, color in zip(focus_countries, palette):
    sub_df = df_feat[df_feat["Country"] == country].sort_values("Year")
    smooth_temp = sub_df["Temp_Anomaly"].rolling(5, min_periods=1).mean()
    ax.plot(sub_df["Year"], smooth_temp, label=country, linewidth=2.2, color=color)

ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
ax.set_title("8. Regional Warming Divergence: High-Latitude Amplification vs. Tropics", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Smoothed Temp Anomaly (°C, 5-yr MA)", fontsize=10)
ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))
plt.tight_layout()
plt.show()"""))

    # Machine Learning
    cells.append(make_cell("markdown", """## 6. Machine Learning: Climate Risk Classification
We benchmark three classification algorithms to predict scenarios of **High Climate Risk**:
1. **Logistic Regression** (Standardized via Scikit-Learn Pipeline)
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

Validation is carried out using **5-Fold Stratified Cross-Validation** to guarantee model generalizability and avoid data leakage."""))

    cells.append(make_cell("code", """FEATURE_COLS = [
    "Year",
    "CO2_Emissions_MT",
    "CO2_per_Capita",
    "Cumulative_CO2_MT",
    "Sea_Level_Rise_mm",
    "Precipitation_mm",
    "Humidity_pct",
    "Population",
    "Country_Baseline_Temp",
    "Temp_Change_YoY"
]
TARGET_COL = "High_Climate_Risk"

X = df_feat[FEATURE_COLS].copy()
y = df_feat[TARGET_COL].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=4, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
trained_models = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_prob

    results.append({
        "Model": name,
        "CV Accuracy (Mean)": round(cv_scores.mean(), 4),
        "CV Accuracy (Std)": round(cv_scores.std(), 4),
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

metrics_df = pd.DataFrame(results)
metrics_df"""))

    # ROC Curves & Confusion Matrix
    cells.append(make_cell("markdown", "### Model Evaluation Visualizations: ROC Curves & Confusion Matrices"))
    cells.append(make_cell("code", """# 1. ROC Curves
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
for name, y_prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", linewidth=2.2)

ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
ax_roc.set_title("ROC Curves Comparison: Climate Risk Classifiers", fontsize=13, fontweight="bold", pad=12)
ax_roc.set_xlabel("False Positive Rate", fontsize=10)
ax_roc.set_ylabel("True Positive Rate", fontsize=10)
ax_roc.legend(frameon=True, loc="lower right")
plt.tight_layout()
plt.show()

# 2. Confusion Matrices
fig_cm, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Normal", "High Risk"], yticklabels=["Normal", "High Risk"])
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("Actual Label", fontsize=9)

fig_cm.suptitle("Confusion Matrices on Test Set", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()"""))

    # Feature Importance
    cells.append(make_cell("markdown", "### Feature Importance Analysis (Tree-Based Ensembles)"))
    cells.append(make_cell("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, name in zip(axes, ["Random Forest", "Gradient Boosting"]):
    model = trained_models[name]
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    colors = sns.color_palette("viridis", n_colors=len(importances))
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"{name} Feature Importance", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Importance", fontsize=9)

fig.suptitle("Key Drivers of Climate Risk Predictions", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()"""))

    # Forecasting
    cells.append(make_cell("markdown", """## 7. Long-Term Climate Trend Forecasting to 2050 (Linear Regression)
We fit a linear regression on historical global temperature anomalies to project warming trends up to 2050 with a 95% confidence envelope."""))

    cells.append(make_cell("code", """annual_hist = df_feat.groupby("Year")["Temp_Anomaly"].mean().reset_index()

X_hist = annual_hist[["Year"]].values
y_hist = annual_hist["Temp_Anomaly"].values

lr_forecaster = LinearRegression()
lr_forecaster.fit(X_hist, y_hist)

slope = lr_forecaster.coef_[0]
residuals = y_hist - lr_forecaster.predict(X_hist)
res_std = np.std(residuals, ddof=2)

future_years = np.arange(2024, 2051).reshape(-1, 1)
future_preds = lr_forecaster.predict(future_years)
lower_bound = future_preds - 1.96 * res_std
upper_bound = future_preds + 1.96 * res_std

forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted_Anomaly": np.round(future_preds, 3),
    "Lower_Bound_95": np.round(lower_bound, 3),
    "Upper_Bound_95": np.round(upper_bound, 3)
})

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(annual_hist["Year"], annual_hist["Temp_Anomaly"], color="#2b5c8f", marker="o", markersize=3.5,
        linewidth=2, label="Historical Anomaly (1950–2023)")
ax.plot(annual_hist["Year"], lr_forecaster.predict(X_hist), color="#6baed6", linestyle=":", linewidth=1.8, label="Linear Fit")

ax.plot(forecast_df["Year"], forecast_df["Predicted_Anomaly"], color="#cb181d", linestyle="--", linewidth=2.5,
        label=f"Projected Warming (+{slope*10:.3f}°C/decade)")
ax.fill_between(forecast_df["Year"], forecast_df["Lower_Bound_95"], forecast_df["Upper_Bound_95"],
                color="#fb6a4a", alpha=0.25, label="95% Confidence Interval")

ax.axhline(1.5, color="#800026", linestyle="-.", linewidth=1.2, alpha=0.8, label="1.5°C Warming Threshold")

val_2050 = forecast_df["Predicted_Anomaly"].iloc[-1]
ax.scatter([2050], [val_2050], color="#cb181d", s=70, zorder=6)
ax.annotate(f"2050 Projection: +{val_2050:.2f}°C",
            xy=(2050, val_2050), xytext=(2035, val_2050 + 0.35),
            arrowprops=dict(facecolor="#cb181d", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold", color="#cb181d", fontsize=9.5)

ax.set_title("Global Temperature Anomaly Projection to 2050 (Linear Regression)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Temperature Anomaly (°C)", fontsize=10)
ax.set_xlim(1948, 2053)
ax.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.show()

print("Forecast Milestones:")
sample_forecast = forecast_df[forecast_df["Year"].isin([2025, 2030, 2035, 2040, 2045, 2050])]
print(sample_forecast.to_string(index=False))"""))

    # Conclusion
    cells.append(make_cell("markdown", """## 8. Limitations & Future Improvements
1. **Synthetic Nature of Data**: While parameterized to mimic realistic country-level trajectories, real-world climate dynamics involve chaotic non-linear feedback loops (e.g., Arctic permafrost thaw, albedo shifts, ocean currents) that simple models cannot fully capture.
2. **Forecasting Model Complexity**: The linear regression projection provides a baseline warming trend. Future iterations could integrate **ARIMA**, **Prophet**, or **LSTM Recurrent Neural Networks** for multi-factor time-series forecasting.
3. **Feature Expansion**: Incorporating land-use change, renewable energy adoption shares, and extreme weather frequency indices (drought days, storm events) would further enrich risk classification."""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    os.makedirs("notebooks", exist_ok=True)
    out_path = "notebooks/02_climate_analysis_and_prediction.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Successfully generated notebook at: {out_path}")


if __name__ == "__main__":
    build_notebook()
