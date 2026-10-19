"""
Data Generation Module for Climate Change Analysis.
Generates a realistic synthetic dataset representing 30 countries from 1950 to 2023 (2,220 rows)
with non-linear climate trends, realistic geographical baselines, and simulated missing values.
"""

import os
import numpy as np
import pandas as pd

# List of 30 diverse countries across 6 continents
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


def generate_climate_dataset(start_year: int = 1950, end_year: int = 2023, seed: int = 42, missing_rate: float = 0.03) -> pd.DataFrame:
    """
    Generate synthetic realistic climate data for 30 countries across specified years.
    Returns a pandas DataFrame with 2,220 rows (74 years * 30 countries).
    """
    np.random.seed(seed)
    years = list(range(start_year, end_year + 1))
    records = []

    for country, cfg in COUNTRIES_CONFIG.items():
        base_temp = cfg["base_temp"]
        warming_rate = cfg["warming_rate"]
        base_pop = cfg["base_pop"]
        pop_growth = cfg["pop_growth"]
        base_co2 = cfg["co2_growth"]
        base_co2_val = cfg["base_co2"]
        precip_base = cfg["precip"]
        humidity_base = cfg["humidity"]

        for idx, year in enumerate(years):
            t = year - start_year  # 0 to 73

            # 1. Non-linear global warming trend (accelerating post 1980)
            # Baseline warming: slow pre-1980, sharp increase post-1980
            accel_factor = 0.008 * (t ** 1.35) if t < 30 else 0.008 * (30 ** 1.35) + 0.032 * ((t - 30) ** 1.15)
            warming_anomaly = accel_factor * warming_rate
            # Solar/ENSO cycle oscillations + Gaussian natural variability
            enso_cycle = 0.22 * np.sin(2 * np.pi * t / 5.2) + 0.15 * np.cos(2 * np.pi * t / 11.0)
            noise_temp = np.random.normal(0, 0.28)
            avg_temp = round(base_temp + warming_anomaly + enso_cycle + noise_temp, 2)

            # 2. Population growth with logistic moderation in modern era
            current_pop = int(base_pop * np.exp(pop_growth * t * (1 - 0.003 * t)))

            # 3. Non-linear CO2 emissions (MT)
            # High emitters peak/plateau recently; developing countries scale exponentially
            if country in ["United States", "Germany", "United Kingdom", "France"]:
                co2_trend = base_co2_val * (1 + 0.025 * t - 0.0003 * (t ** 2))
                if year > 2005:
                    co2_trend *= (1 - 0.012 * (year - 2005))
            elif country in ["China", "India", "Vietnam"]:
                # Rapid industrial surge after 1980
                boost = 1 + np.exp((year - 1990) / 10.0) / (1 + np.exp((year - 1990) / 10.0)) * 3.5
                co2_trend = base_co2_val * boost * (1 + 0.015 * t)
            else:
                co2_trend = base_co2_val * np.exp(cfg["co2_growth"] * t * 0.9)
            co2_emissions = round(max(5.0, co2_trend + np.random.normal(0, 0.05 * co2_trend)), 2)

            # 4. Sea level rise (mm): global cumulative rise starting around 0 mm in 1950,
            # reaching ~210-240 mm by 2023 with quadratic acceleration
            sea_level = round(0.9 * t + 0.028 * (t ** 2) + np.random.normal(0, 2.5), 2)

            # 5. Precipitation (mm): seasonal/interannual volatility, droughts and storms
            precip_trend = precip_base * (1 + 0.0008 * warming_anomaly + np.random.normal(0, 0.09))
            precip_mm = round(max(10.0, precip_trend), 1)

            # 6. Humidity (%): physical relationship with temperature and precipitation
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

    df = pd.DataFrame(records)

    # Introduce random missing values (~3%) into numerical climate features
    cols_with_missing = ["Average_Temperature_C", "CO2_Emissions_MT", "Precipitation_mm", "Humidity_pct"]
    for col in cols_with_missing:
        mask = np.random.rand(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    return df


def save_dataset(df: pd.DataFrame, file_path: str = "data/climate_data_raw.csv") -> None:
    """Save dataset to CSV file ensuring parent directories exist."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved dataset with {len(df)} rows and {len(df.columns)} columns to {file_path}")


if __name__ == "__main__":
    df = generate_climate_dataset()
    save_dataset(df, "data/climate_data_raw.csv")
