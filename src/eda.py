"""
Exploratory Data Analysis (EDA) Module for Climate Change Analysis.
Generates 8 visually appealing, publication-quality charts using Matplotlib and Seaborn.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set global aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_01_global_temp_trend(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 1: Global Average Temperature Anomaly Trend (1950-2023)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    annual_global = df.groupby("Year").agg({
        "Temp_Anomaly": ["mean", "std"]
    }).reset_index()
    annual_global.columns = ["Year", "Mean_Anomaly", "Std_Anomaly"]
    annual_global["Rolling_5yr"] = annual_global["Mean_Anomaly"].rolling(5, min_periods=1).mean()

    # Bar chart of anomalies
    colors = ["#d73027" if val >= 0 else "#4575b4" for val in annual_global["Mean_Anomaly"]]
    ax.bar(annual_global["Year"], annual_global["Mean_Anomaly"], color=colors, alpha=0.6, width=0.8, label="Annual Anomaly")
    
    # 5-year rolling trendline
    ax.plot(annual_global["Year"], annual_global["Rolling_5yr"], color="#800026", linewidth=2.5, label="5-Year Moving Avg")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    
    # Annotate post-1980 acceleration
    ax.annotate("Accelerated Post-1980 Warming", xy=(1985, 0.4), xytext=(1960, 1.2),
                arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.2, headwidth=6),
                fontsize=9, fontweight="bold", color="#333333")

    ax.set_title("1. Global Average Temperature Anomaly Trend (1950–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Temperature Anomaly (°C relative to 1950-1980)", fontsize=10, labelpad=8)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "01_global_temp_trend.png"))
    return fig


def plot_02_top_emitters(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 2: Top 10 Cumulative CO2 Emitting Countries (1950-2023)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    top_emitters = df.groupby("Country")["CO2_Emissions_MT"].sum().nlargest(10).sort_values()
    
    colors = sns.color_palette("rocket", n_colors=len(top_emitters))
    bars = ax.barh(top_emitters.index, top_emitters.values / 1e3, color=colors, edgecolor="none", height=0.65)
    
    # Value annotations on bar tips
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2, f"{width:.1f} GT",
                va="center", ha="left", fontsize=8.5, color="#222222")

    ax.set_title("2. Top 10 Cumulative CO2 Emitting Countries (Gigatons, 1950–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cumulative CO2 Emissions (Gigatons, GT)", fontsize=10, labelpad=8)
    ax.set_ylabel("Country", fontsize=10)
    ax.set_xlim(0, (top_emitters.max() / 1e3) * 1.15)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "02_top_co2_emitters.png"))
    return fig


def plot_03_correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 3: Correlation Heatmap of Climate Indicators."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
    
    feature_cols = [
        "Average_Temperature_C", "Temp_Anomaly", "CO2_Emissions_MT",
        "CO2_per_Capita", "Sea_Level_Rise_mm", "Precipitation_mm",
        "Humidity_pct", "Population"
    ]
    corr_matrix = df[feature_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                linewidths=0.8, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title("3. Correlation Matrix of Global Climate Indicators", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "03_correlation_heatmap.png"))
    return fig


def plot_04_sea_level_rise(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 4: Cumulative Sea Level Rise Acceleration (1950-2023)."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    slr_data = df.groupby("Year")["Sea_Level_Rise_mm"].mean().reset_index()
    
    ax.plot(slr_data["Year"], slr_data["Sea_Level_Rise_mm"], color="#08519c", linewidth=2.8, label="Global Mean SLR (mm)")
    ax.fill_between(slr_data["Year"], 0, slr_data["Sea_Level_Rise_mm"], color="#9ecae1", alpha=0.35)
    
    # Highlight final value
    final_year = slr_data["Year"].iloc[-1]
    final_slr = slr_data["Sea_Level_Rise_mm"].iloc[-1]
    ax.scatter([final_year], [final_slr], color="#08519c", s=60, zorder=5)
    ax.text(final_year - 6, final_slr + 8, f"{final_slr:.1f} mm\n(2023)", fontweight="bold", color="#08519c", fontsize=9)

    ax.set_title("4. Cumulative Global Mean Sea Level Rise (1950–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Sea Level Rise (mm above 1950 base)", fontsize=10, labelpad=8)
    ax.set_ylim(bottom=-5)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "04_sea_level_rise.png"))
    return fig


def plot_05_temperature_anomaly_distribution(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 5: Shifting Distribution of Temperature Anomalies by Decade."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    df_decade = df.copy()
    df_decade["Decade"] = (df_decade["Year"] // 10) * 10
    df_decade["Decade_Label"] = df_decade["Decade"].astype(str) + "s"
    
    order = sorted(df_decade["Decade_Label"].unique())
    palette = sns.color_palette("YlOrRd", n_colors=len(order))
    
    sns.boxplot(x="Decade_Label", y="Temp_Anomaly", data=df_decade, order=order,
                hue="Decade_Label", legend=False, palette=palette, width=0.55, fliersize=2.5, ax=ax)
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("5. Decadal Shift in Temperature Anomaly Distributions (1950s–2020s)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Decade", fontsize=10, labelpad=8)
    ax.set_ylabel("Temperature Anomaly (°C)", fontsize=10, labelpad=8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "05_temp_anomaly_decades.png"))
    return fig


def plot_06_co2_vs_temp_scatter(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 6: CO2 Emissions vs Temperature Anomaly Scatter."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    
    sample_df = df.copy()
    scatter = ax.scatter(
        sample_df["CO2_per_Capita"],
        sample_df["Temp_Anomaly"],
        c=sample_df["Year"],
        cmap="plasma",
        alpha=0.6,
        s=30,
        edgecolors="none"
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Observation Year", fontsize=9)
    
    # Trend line
    sns.regplot(
        x="CO2_per_Capita", y="Temp_Anomaly", data=sample_df,
        scatter=False, ax=ax, color="#2b2b2b", line_kws={"linewidth": 2, "linestyle": "--"}
    )

    ax.set_title("6. Per Capita CO2 Emissions vs. Temperature Anomaly (Colored by Year)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("CO2 Emissions per Capita (Metric Tons)", fontsize=10, labelpad=8)
    ax.set_ylabel("Temperature Anomaly (°C)", fontsize=10, labelpad=8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "06_co2_vs_temp_scatter.png"))
    return fig


def plot_07_precipitation_trends(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 7: Annual Precipitation Trends and Extreme Fluctuations."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    precip_annual = df.groupby("Year")["Precipitation_mm"].agg(["mean", "std"]).reset_index()
    precip_annual["Rolling_5yr"] = precip_annual["mean"].rolling(5, min_periods=1).mean()
    
    ax.fill_between(
        precip_annual["Year"],
        precip_annual["mean"] - precip_annual["std"],
        precip_annual["mean"] + precip_annual["std"],
        color="#a1d99b", alpha=0.35, label="±1 Std Dev Spread (Regional Variance)"
    )
    ax.plot(precip_annual["Year"], precip_annual["mean"], color="#238b45", linewidth=2.0, alpha=0.8, label="Global Mean Precipitation")
    ax.plot(precip_annual["Year"], precip_annual["Rolling_5yr"], color="#00441b", linewidth=2.5, label="5-Year Moving Avg")

    ax.set_title("7. Global Annual Mean Precipitation & Interannual Fluctuations (1950–2023)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Precipitation (mm / year)", fontsize=10, labelpad=8)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "07_precipitation_trends.png"))
    return fig


def plot_08_multi_country_trajectories(df: pd.DataFrame, output_dir: str = "outputs/figures") -> plt.Figure:
    """Chart 8: Multi-Country Temperature Trajectories (High-Latitude Amplification vs Tropics)."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    
    focus_countries = ["Canada", "Norway", "United States", "China", "India", "Brazil", "Australia"]
    palette = sns.color_palette("tab10", n_colors=len(focus_countries))
    
    for country, color in zip(focus_countries, palette):
        country_df = df[df["Country"] == country].sort_values("Year")
        # 5-year rolling average for smoothing lines
        smooth_temp = country_df["Temp_Anomaly"].rolling(5, min_periods=1).mean()
        ax.plot(country_df["Year"], smooth_temp, label=country, linewidth=2.2, color=color)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title("8. Regional Temperature Anomaly Trajectories: Arctic Amplification vs Tropics", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10, labelpad=8)
    ax.set_ylabel("Smoothed Temp Anomaly (°C, 5-yr MA)", fontsize=10, labelpad=8)
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "08_multi_country_trajectories.png"))
    return fig


def run_all_eda(df: pd.DataFrame, output_dir: str = "outputs/figures") -> None:
    """Generates and saves all 8 EDA charts."""
    print("Generating 8 EDA charts...")
    plot_01_global_temp_trend(df, output_dir)
    plot_02_top_emitters(df, output_dir)
    plot_03_correlation_heatmap(df, output_dir)
    plot_04_sea_level_rise(df, output_dir)
    plot_05_temperature_anomaly_distribution(df, output_dir)
    plot_06_co2_vs_temp_scatter(df, output_dir)
    plot_07_precipitation_trends(df, output_dir)
    plot_08_multi_country_trajectories(df, output_dir)
    print(f"All 8 charts successfully generated and saved to '{output_dir}'.")


if __name__ == "__main__":
    df = pd.read_csv("data/climate_data_cleaned.csv")
    run_all_eda(df)
