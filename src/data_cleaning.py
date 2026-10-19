"""
Data Cleaning and Feature Engineering Module.
Performs country-specific median imputation and constructs predictive features
including CO2_per_Capita, Temp_Change_YoY, Temp_Anomaly, and High_Climate_Risk target.
"""

import os
import pandas as pd
import numpy as np


def clean_climate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans climate dataset by imputing missing numerical values using country-specific medians.
    Ensures sort order by Country and Year.
    """
    df_clean = df.copy()
    df_clean = df_clean.sort_values(by=["Country", "Year"]).reset_index(drop=True)

    numeric_cols = [
        "Average_Temperature_C",
        "CO2_Emissions_MT",
        "Sea_Level_Rise_mm",
        "Precipitation_mm",
        "Humidity_pct",
        "Population"
    ]

    for col in numeric_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            # Country-specific median imputation
            df_clean[col] = df_clean.groupby("Country")[col].transform(
                lambda grp: grp.fillna(grp.median())
            )
            # Global median fallback just in case
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-relevant climate indicators and target variable:
    - CO2_per_Capita: Metric tons of CO2 emitted per person.
    - Country_Baseline_Temp: Historical baseline average (1950-1980 reference period).
    - Temp_Anomaly: Deviation from reference baseline.
    - Temp_Change_YoY: Year-over-Year change in temperature per country.
    - Temp_5yr_MA: 5-year rolling moving average temperature.
    - Cumulative_CO2_MT: Cumulative CO2 emissions by country since 1950.
    - High_Climate_Risk: Categorical binary target (1 if Temp_Anomaly > 75th percentile, else 0).
    """
    df_feat = df_clean.copy()

    # 1. CO2 per Capita (Metric Tons per person)
    df_feat["CO2_per_Capita"] = (df_feat["CO2_Emissions_MT"] * 1_000_000) / df_feat["Population"]

    # 2. Historical Baseline Temperature (1950-1980 period)
    baseline_temp = (
        df_feat[df_feat["Year"] <= 1980]
        .groupby("Country")["Average_Temperature_C"]
        .mean()
        .to_dict()
    )
    df_feat["Country_Baseline_Temp"] = df_feat["Country"].map(baseline_temp)

    # 3. Temperature Anomaly (difference from country 1950-1980 reference)
    df_feat["Temp_Anomaly"] = round(df_feat["Average_Temperature_C"] - df_feat["Country_Baseline_Temp"], 3)

    # 4. Year-over-Year Temperature Change
    df_feat["Temp_Change_YoY"] = df_feat.groupby("Country")["Average_Temperature_C"].diff().fillna(0)

    # 5. 5-year Rolling Moving Average
    df_feat["Temp_5yr_MA"] = df_feat.groupby("Country")["Average_Temperature_C"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    # 6. Cumulative CO2
    df_feat["Cumulative_CO2_MT"] = df_feat.groupby("Country")["CO2_Emissions_MT"].cumsum()

    # 7. High_Climate_Risk Target: Defined as Temp_Anomaly exceeding 75th percentile
    anomaly_p75 = df_feat["Temp_Anomaly"].quantile(0.75)
    df_feat["High_Climate_Risk"] = (df_feat["Temp_Anomaly"] > anomaly_p75).astype(int)

    return df_feat


def process_and_save(raw_csv: str = "data/climate_data_raw.csv", cleaned_csv: str = "data/climate_data_cleaned.csv") -> pd.DataFrame:
    """Loads raw dataset, cleans it, engineers features, and saves cleaned dataset."""
    df_raw = pd.read_csv(raw_csv)
    df_clean = clean_climate_data(df_raw)
    df_featured = engineer_features(df_clean)
    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)
    df_featured.to_csv(cleaned_csv, index=False)
    print(f"Cleaned and featured data saved to {cleaned_csv} ({len(df_featured)} rows, {len(df_featured.columns)} columns)")
    return df_featured


if __name__ == "__main__":
    process_and_save()
