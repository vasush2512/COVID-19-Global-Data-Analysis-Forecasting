"""
visualizations.py
─────────────────
Generates the 6-chart Tableau-equivalent dashboard using Matplotlib & Seaborn.

Charts
------
1. Global Daily New Cases (7-day avg) — line chart
2. Case Fatality Rate by Country — horizontal bar chart
3. Regional Burden (Total Cases) — pie / donut chart
4. Top-10 Countries: Cumulative Cases — bar race snapshot
5. ARIMA Forecast with Confidence Interval — line + shaded band
6. Monthly CFR Trend (Global) — area + line combo

All charts are saved to outputs/ and optionally shown interactively.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import OUTPUT_DIR, FIG_DPI, STYLE, PALETTE

logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
try:
    plt.style.use(STYLE)
except OSError:
    plt.style.use("seaborn-whitegrid")

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Global Daily New Cases (7-day rolling avg)
# ─────────────────────────────────────────────────────────────────────────────

def chart_global_trend(df: pd.DataFrame) -> str:
    """Line chart: global 7-day avg new cases over time."""
    daily = (
        df.groupby("Date_reported")["New_cases"]
        .sum()
        .reset_index()
        .sort_values("Date_reported")
    )
    daily["Avg7"] = daily["New_cases"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(daily["Date_reported"], daily["New_cases"],
                    alpha=0.15, color=COLORS[0], label="Daily new cases")
    ax.plot(daily["Date_reported"], daily["Avg7"],
            color=COLORS[0], linewidth=2, label="7-day rolling avg")

    ax.set_title("Global Daily New COVID-19 Cases", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("New Cases")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.legend(frameon=True)
    ax.margins(x=0.01)
    fig.tight_layout()
    return _save(fig, "chart1_global_trend.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Case Fatality Rate by Country (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────

def chart_cfr_by_country(cfr_df: pd.DataFrame, top_n: int = 20) -> str:
    """Horizontal bar: CFR % for top-N countries."""
    data = cfr_df.head(top_n).sort_values("CFR_pct")

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        data["Country"], data["CFR_pct"],
        color=sns.color_palette("Reds_r", top_n),
        edgecolor="white", linewidth=0.4,
    )
    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}%", va="center", fontsize=8)

    ax.set_title("Case Fatality Rate by Country (Top 20)", fontsize=15, fontweight="bold")
    ax.set_xlabel("CFR (%)")
    ax.set_xlim(0, data["CFR_pct"].max() * 1.2)
    fig.tight_layout()
    return _save(fig, "chart2_cfr_by_country.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Regional Burden (Donut Chart)
# ─────────────────────────────────────────────────────────────────────────────

def chart_regional_burden(region_df: pd.DataFrame) -> str:
    """Donut chart: total cases by WHO region."""
    fig, ax = plt.subplots(figsize=(9, 7))

    region_labels = {
        "AMRO": "Americas", "EURO": "Europe", "SEARO": "South-East Asia",
        "WPRO": "Western Pacific", "AFRO": "Africa", "EMRO": "E. Mediterranean",
    }
    labels = [region_labels.get(r, r) for r in region_df["WHO_region"]]
    sizes  = region_df["Total_cases"].values
    colors = sns.color_palette("tab10", len(sizes))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_fontsize(9)

    # Centre text
    total = sizes.sum()
    ax.text(0, 0, f"{total/1e9:.2f}B\nTotal Cases",
            ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_title("COVID-19 Cases by WHO Region", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "chart3_regional_burden.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Top-10 Countries: Cumulative Cases (grouped bar snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def chart_top10_cumulative(df: pd.DataFrame, top_n: int = 10) -> str:
    """Bar chart: top-10 countries by total cumulative cases."""
    latest = (
        df.sort_values("Date_reported")
          .groupby("Country")[["Cumulative_cases", "Cumulative_deaths", "WHO_region"]]
          .last()
          .reset_index()
          .nlargest(top_n, "Cumulative_cases")
          .sort_values("Cumulative_cases")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette(PALETTE, top_n)
    bars = ax.barh(latest["Country"], latest["Cumulative_cases"],
                   color=palette, edgecolor="white", linewidth=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{w/1e6:.1f}M", va="center", fontsize=9)

    ax.set_title(f"Top {top_n} Countries by Cumulative COVID-19 Cases", fontsize=15, fontweight="bold")
    ax.set_xlabel("Cumulative Cases")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.set_xlim(0, latest["Cumulative_cases"].max() * 1.15)
    fig.tight_layout()
    return _save(fig, "chart4_top10_cumulative.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 — ARIMA Forecast with Confidence Interval
# ─────────────────────────────────────────────────────────────────────────────

def chart_arima_forecast(arima_results: dict, lookback_days: int = 120) -> str:
    """
    Line chart showing:
      - Historical 7-day avg (last `lookback_days`)
      - Test-set actual vs predicted
      - 30-day forecast with 95 % CI shaded band
    """
    train = arima_results["train_series"]
    test  = arima_results["test_series"]
    preds = arima_results["test_predictions"]
    fc_dates  = arima_results["forecast_dates"]
    fc_values = arima_results["forecast_values"]
    fc_lower  = arima_results["forecast_lower"]
    fc_upper  = arima_results["forecast_upper"]
    metrics   = arima_results["metrics"]
    country   = arima_results["country"]

    hist = pd.concat([train, test]).iloc[-lookback_days:]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical
    ax.plot(hist.index, hist.values,
            color=COLORS[0], linewidth=1.8, label="Historical (7-day avg)")

    # Test predictions
    ax.plot(preds.index, preds.values,
            color=COLORS[1], linewidth=1.5, linestyle="--", label="ARIMA test-fit")

    # Forecast
    ax.plot(fc_dates, fc_values,
            color=COLORS[2], linewidth=2.2, label=f"{len(fc_dates)}-day forecast")
    ax.fill_between(fc_dates, fc_lower, fc_upper,
                    color=COLORS[2], alpha=0.18, label="95 % Confidence Interval")

    # Vertical divider at forecast start
    ax.axvline(fc_dates[0], color="grey", linestyle=":", linewidth=1)
    ax.text(fc_dates[0], ax.get_ylim()[1] * 0.92, " Forecast\n start",
            fontsize=8, color="grey")

    ax.set_title(
        f"ARIMA{arima_results['order']} 30-Day Case Forecast — {country}\n"
        f"Test MAPE: {metrics['MAPE']}%  |  RMSE: {metrics['RMSE']:,.0f}  |  MAE: {metrics['MAE']:,.0f}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily New Cases (7-day avg)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.legend(frameon=True, loc="upper left")
    ax.margins(x=0.01)
    fig.tight_layout()
    return _save(fig, "chart5_arima_forecast.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6 — Monthly CFR Trend (Global)
# ─────────────────────────────────────────────────────────────────────────────

def chart_monthly_cfr(monthly_df: pd.DataFrame) -> str:
    """
    Dual-axis area+line combo:
      - Left  axis: monthly new cases (area)
      - Right axis: monthly CFR % (line with markers)
    """
    df = monthly_df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"])
    df = df.sort_values("Month_dt").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax1.fill_between(df["Month_dt"], df["New_cases"],
                     alpha=0.25, color=COLORS[0])
    ax1.plot(df["Month_dt"], df["New_cases"],
             color=COLORS[0], linewidth=1.5, label="Monthly new cases")
    ax1.set_ylabel("Monthly New Cases", color=COLORS[0])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    )

    ax2.plot(df["Month_dt"], df["CFR_pct"],
             color=COLORS[3], linewidth=2.2, marker="o", markersize=3, label="Monthly CFR %")
    ax2.set_ylabel("Case Fatality Rate (%)", color=COLORS[3])
    ax2.tick_params(axis="y", labelcolor=COLORS[3])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    ax1.set_title("Global Monthly New Cases vs Case Fatality Rate",
                  fontsize=15, fontweight="bold")
    ax1.set_xlabel("Month")
    fig.tight_layout()
    return _save(fig, "chart6_monthly_cfr.png")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard composer — all 6 charts in one figure
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(
    df: pd.DataFrame,
    cfr_df: pd.DataFrame,
    region_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    arima_results: dict,
) -> str:
    """
    Assemble all 6 charts into a single 3×2 dashboard PNG.
    Saves individual charts first, then composes the dashboard.
    """
    logger.info("Generating individual charts …")
    chart_global_trend(df)
    chart_cfr_by_country(cfr_df)
    chart_regional_burden(region_df)
    chart_top10_cumulative(df)
    chart_arima_forecast(arima_results)
    chart_monthly_cfr(monthly_df)

    logger.info("Composing dashboard …")
    from matplotlib.image import imread

    chart_files = [
        "chart1_global_trend.png",
        "chart2_cfr_by_country.png",
        "chart3_regional_burden.png",
        "chart4_top10_cumulative.png",
        "chart5_arima_forecast.png",
        "chart6_monthly_cfr.png",
    ]
    titles = [
        "① Global Daily New Cases",
        "② CFR by Country",
        "③ Regional Burden",
        "④ Top-10 Cumulative Cases",
        "⑤ ARIMA 30-Day Forecast",
        "⑥ Monthly CFR Trend",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(22, 24))
    fig.suptitle(
        "COVID-19 Global Data Analysis Dashboard",
        fontsize=22, fontweight="bold", y=1.005,
    )

    for ax, fname, title in zip(axes.flat, chart_files, titles):
        path = os.path.join(OUTPUT_DIR, fname)
        img  = imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
        ax.axis("off")

    plt.subplots_adjust(hspace=0.08, wspace=0.04)
    dashboard_path = _save(fig, "dashboard_all_charts.png")
    logger.info("Dashboard saved → %s", dashboard_path)
    return dashboard_path


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from etl_pipeline import load_clean_data
    from analysis import cfr_by_country, regional_burden, monthly_global_trend
    from arima_model import run_arima_pipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
    df         = load_clean_data()
    cfr_df     = cfr_by_country(df)
    region_df  = regional_burden(df)
    monthly_df = monthly_global_trend(df)
    arima_res  = run_arima_pipeline(df)

    build_dashboard(df, cfr_df, region_df, monthly_df, arima_res)
    print("All charts saved to:", OUTPUT_DIR)
