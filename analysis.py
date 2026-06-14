"""
analysis.py
───────────
Derives key public-health insights from clean WHO COVID-19 data:

  • Case Fatality Rate (CFR) by country
  • Top / bottom countries by peak daily cases
  • Regional burden comparison
  • Monthly trend summaries
  • Wave detection (local maxima)

All functions accept a clean DataFrame (from etl_pipeline) and return
structured DataFrames ready for visualisation or reporting.
"""

import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CFR Analysis
# ─────────────────────────────────────────────────────────────────────────────

def cfr_by_country(df: pd.DataFrame, min_cases: int = 50_000) -> pd.DataFrame:
    """
    Compute final Case Fatality Rate per country.

    Parameters
    ----------
    df        : Clean WHO DataFrame.
    min_cases : Only include countries with at least this many cumulative cases.

    Returns
    -------
    DataFrame sorted by CFR descending with columns:
        Country, Total_cases, Total_deaths, CFR_pct
    """
    latest = (
        df.sort_values("Date_reported")
          .groupby("Country")
          .last()
          .reset_index()
    )
    latest = latest[latest["Cumulative_cases"] >= min_cases]
    latest["CFR_pct"] = (
        latest["Cumulative_deaths"] / latest["Cumulative_cases"] * 100
    ).round(3)

    result = (
        latest[["Country", "WHO_region", "Cumulative_cases", "Cumulative_deaths", "CFR_pct"]]
        .rename(columns={
            "Cumulative_cases":  "Total_cases",
            "Cumulative_deaths": "Total_deaths",
        })
        .sort_values("CFR_pct", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("CFR computed for %d countries.", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Peak Case Analysis
# ─────────────────────────────────────────────────────────────────────────────

def peak_cases_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find each country's single highest daily new-case count and when it occurred.

    Returns
    -------
    DataFrame with: Country, Peak_date, Peak_new_cases, WHO_region
    """
    idx = df.groupby("Country")["New_cases"].idxmax()
    peak_df = df.loc[idx, ["Country", "Date_reported", "New_cases", "WHO_region"]].copy()
    peak_df = peak_df.rename(columns={
        "Date_reported": "Peak_date",
        "New_cases":     "Peak_new_cases",
    })
    peak_df = peak_df.sort_values("Peak_new_cases", ascending=False).reset_index(drop=True)
    logger.info("Peak cases identified for %d countries.", len(peak_df))
    return peak_df


# ─────────────────────────────────────────────────────────────────────────────
# Regional Burden
# ─────────────────────────────────────────────────────────────────────────────

def regional_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total cases, deaths and average CFR by WHO region.

    Returns
    -------
    DataFrame: WHO_region, Total_cases, Total_deaths, Avg_CFR_pct, Countries
    """
    latest = (
        df.sort_values("Date_reported")
          .groupby(["Country", "WHO_region"])
          .last()
          .reset_index()
    )
    region_df = (
        latest.groupby("WHO_region")
        .agg(
            Total_cases  =("Cumulative_cases",  "sum"),
            Total_deaths =("Cumulative_deaths", "sum"),
            Countries    =("Country",           "nunique"),
        )
        .reset_index()
    )
    region_df["Avg_CFR_pct"] = (
        region_df["Total_deaths"] / region_df["Total_cases"] * 100
    ).round(3)
    region_df = region_df.sort_values("Total_cases", ascending=False).reset_index(drop=True)
    return region_df


# ─────────────────────────────────────────────────────────────────────────────
# Monthly Trends
# ─────────────────────────────────────────────────────────────────────────────

def monthly_global_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Global monthly new-case and new-death totals.

    Returns
    -------
    DataFrame: Month (Period), New_cases, New_deaths, CFR_pct
    """
    monthly = (
        df.groupby("Month")
        .agg(
            New_cases  =("New_cases",   "sum"),
            New_deaths =("New_deaths",  "sum"),
        )
        .reset_index()
    )
    monthly["CFR_pct"] = (
        monthly["New_deaths"] / monthly["New_cases"].replace(0, np.nan) * 100
    ).round(3)
    monthly = monthly.sort_values("Month").reset_index(drop=True)
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Wave Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_waves(df: pd.DataFrame, country: str, prominence: float = 0.15) -> pd.DataFrame:
    """
    Detect epidemic waves for a given country using scipy peak-finding.

    Parameters
    ----------
    df         : Clean DataFrame.
    country    : Country name string.
    prominence : Relative peak prominence threshold (fraction of max).

    Returns
    -------
    DataFrame with columns: Peak_date, Peak_cases, Wave_number
    """
    cdf = (
        df[df["Country"] == country]
        .sort_values("Date_reported")
        .set_index("Date_reported")
    )
    series = cdf["Cases_7day_avg"].fillna(0).values
    prom   = prominence * series.max()
    peaks, props = find_peaks(series, prominence=prom, distance=30)

    if len(peaks) == 0:
        logger.warning("No waves detected for %s.", country)
        return pd.DataFrame(columns=["Peak_date", "Peak_cases", "Wave_number"])

    dates = cdf.index[peaks]
    wave_df = pd.DataFrame({
        "Peak_date":   dates,
        "Peak_cases":  series[peaks].astype(int),
        "Wave_number": range(1, len(peaks) + 1),
    })
    logger.info("Detected %d wave(s) for %s.", len(wave_df), country)
    return wave_df


# ─────────────────────────────────────────────────────────────────────────────
# Recovery Proxy (estimated)
# ─────────────────────────────────────────────────────────────────────────────

def recovery_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHO stopped publishing recovery data in 2021.
    Estimate active cases as:
        Active ≈ Cumulative_cases - Cumulative_deaths - assumed_recovered
    where assumed_recovered = cases shifted forward 14 days.

    Returns
    -------
    DataFrame with Country, Date_reported, Estimated_active, Estimated_recovered
    """
    records = []
    for country, grp in df.groupby("Country"):
        g = grp.sort_values("Date_reported").copy()
        g["Recovered_14d"] = g["Cumulative_cases"].shift(14).fillna(0)
        g["Estimated_recovered"] = (
            g["Recovered_14d"] - g["Cumulative_deaths"]
        ).clip(lower=0)
        g["Estimated_active"] = (
            g["Cumulative_cases"] - g["Estimated_recovered"] - g["Cumulative_deaths"]
        ).clip(lower=0)
        records.append(g)

    return pd.concat(records, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Quick summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a high-level text summary to stdout."""
    total_cases  = df.groupby("Country")["Cumulative_cases"].last().sum()
    total_deaths = df.groupby("Country")["Cumulative_deaths"].last().sum()
    global_cfr   = total_deaths / total_cases * 100

    print("\n" + "═" * 55)
    print("   COVID-19 GLOBAL SUMMARY")
    print("═" * 55)
    print(f"  Countries analysed : {df['Country'].nunique()}")
    print(f"  Date range         : {df['Date_reported'].min().date()} → {df['Date_reported'].max().date()}")
    print(f"  Total records      : {len(df):,}")
    print(f"  Total cases        : {total_cases:,.0f}")
    print(f"  Total deaths       : {total_deaths:,.0f}")
    print(f"  Global CFR         : {global_cfr:.2f} %")
    print("═" * 55 + "\n")

    cfr_df = cfr_by_country(df)
    print("Top 10 countries by CFR:")
    print(cfr_df.head(10)[["Country", "Total_cases", "CFR_pct"]].to_string(index=False))

    print("\nRegional burden:")
    print(regional_burden(df).to_string(index=False))
