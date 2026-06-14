"""
main.py
-------
Orchestrates the full COVID-19 analysis pipeline end-to-end:

  Step 1 - ETL       : Fetch / generate -> clean -> save data
  Step 2 - Analysis  : CFR, peaks, regional burden, monthly trends
  Step 3 - Forecasting: ARIMA model + evaluation metrics
  Step 4 - Visuals   : 6-chart dashboard

Run:  python main.py
"""

import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    t0 = time.time()

    print("\n" + "=" * 60)
    print("   COVID-19 GLOBAL DATA ANALYSIS & FORECASTING PIPELINE")
    print("=" * 60 + "\n")

    # --------------------------------------------------------------
    # STEP 1 - ETL
    # --------------------------------------------------------------
    print("[STEP 1] ETL Pipeline")
    print("-" * 40)
    from etl_pipeline import run_etl
    df = run_etl()
    print(f"    [OK] Records loaded  : {len(df):,}")
    print(f"    [OK] Countries       : {df['Country'].nunique()}")
    print(f"    [OK] Date range      : {df['Date_reported'].min().date()} -> {df['Date_reported'].max().date()}")

    # --------------------------------------------------------------
    # STEP 2 - Analysis
    # --------------------------------------------------------------
    print("\n[STEP 2] Exploratory Analysis")
    print("-" * 40)
    from analysis import (
        cfr_by_country, peak_cases_by_country,
        regional_burden, monthly_global_trend,
        detect_waves, print_summary,
    )

    cfr_df     = cfr_by_country(df)
    peak_df    = peak_cases_by_country(df)
    region_df  = regional_burden(df)
    monthly_df = monthly_global_trend(df)

    print_summary(df)

    print("  Top 5 countries by CFR:")
    for _, row in cfr_df.head(5).iterrows():
        print(f"    - {row['Country']:<40} CFR = {row['CFR_pct']:.2f} %")

    print("\n  Top 5 countries by peak daily cases:")
    for _, row in peak_df.head(5).iterrows():
        print(f"    - {row['Country']:<40} {row['Peak_new_cases']:>10,}  on {row['Peak_date'].date()}")

    print("\n  Regional burden:")
    for _, row in region_df.iterrows():
        print(f"    - {row['WHO_region']:<8}  Cases={row['Total_cases']/1e6:>7.1f}M  "
              f"CFR={row['Avg_CFR_pct']:.2f}%")

    # Wave detection for India
    from config import FORECAST_COUNTRY
    waves = detect_waves(df, FORECAST_COUNTRY)              


    
    print(f"\n  Epidemic waves detected for {FORECAST_COUNTRY}: {len(waves)}")
    if not waves.empty:
        for _, w in waves.iterrows():
            print(f"    Wave {w['Wave_number']}: peak {w['Peak_cases']:,} cases on {w['Peak_date'].date()}")

    # --------------------------------------------------------------
    # STEP 3 - ARIMA Forecasting
    # --------------------------------------------------------------
    print(f"\n[STEP 3] ARIMA Forecasting ({FORECAST_COUNTRY})")
    print("-" * 40)
    from arima_model import run_arima_pipeline
    arima_results = run_arima_pipeline(df)

    m = arima_results["metrics"]
    print(f"    [OK] Best order      : ARIMA{arima_results['order']}")
    print(f"    [OK] MAPE            : {m['MAPE']} %")
    print(f"    [OK] RMSE            : {m['RMSE']:,.1f}")
    print(f"    [OK] MAE             : {m['MAE']:,.1f}")
    print(f"\n    30-day forecast preview:")
    for d, v, lo, hi in zip(
        arima_results["forecast_dates"][:5],
        arima_results["forecast_values"][:5],
        arima_results["forecast_lower"][:5],
        arima_results["forecast_upper"][:5],
    ):
        print(f"      {d.date()}  ->  {v:>9,.0f}  [CI: {lo:,.0f} - {hi:,.0f}]")

    # --------------------------------------------------------------
    # STEP 4 - Visualizations
    # --------------------------------------------------------------
    print("\n[STEP 4] Building 6-Chart Dashboard")
    print("-" * 40)
    from visualizations import build_dashboard
    dashboard_path = build_dashboard(df, cfr_df, region_df, monthly_df, arima_results)
    print(f"    [OK] Dashboard saved -> {dashboard_path}")

    # --------------------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"   [DONE] Pipeline complete in {elapsed:.1f}s")
    print(f"   [DIR]  Outputs saved in: {os.path.abspath('outputs/')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
