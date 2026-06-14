# config.py — Central configuration for the COVID-19 Analysis Project

import os

# ── Data Sources ──────────────────────────────────────────────────────────────
WHO_DATA_URL = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
RAW_FILE     = os.path.join(DATA_DIR, "who_covid19_raw.csv")
CLEAN_FILE   = os.path.join(DATA_DIR, "who_covid19_clean.csv")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")

# ── Countries of interest (50+ covered; subset shown for targeted analysis) ───
TARGET_COUNTRIES = [
    "India", "United States of America", "Brazil", "Germany", "France",
    "United Kingdom", "Italy", "Spain", "Russia", "South Africa",
    "Mexico", "Argentina", "Colombia", "Peru", "Chile",
    "Indonesia", "Philippines", "Vietnam", "Thailand", "Malaysia",
    "Nigeria", "Kenya", "Ethiopia", "Egypt", "Morocco",
    "Japan", "Republic of Korea", "Australia", "Canada", "Turkey",
    "Iran (Islamic Republic of)", "Pakistan", "Bangladesh", "Nepal", "Sri Lanka",
    "Saudi Arabia", "United Arab Emirates", "Israel", "Iraq", "Jordan",
    "Poland", "Ukraine", "Romania", "Netherlands", "Belgium",
    "Portugal", "Sweden", "Norway", "Denmark", "Austria", "Switzerland",
]

# ── ARIMA Forecasting ─────────────────────────────────────────────────────────
FORECAST_COUNTRY  = "India"          # Country to forecast
FORECAST_HORIZON  = 30               # Days ahead
ARIMA_SEASONAL    = False            # Set True to use SARIMA
TRAIN_RATIO       = 0.85             # 85 % train / 15 % test split

# ── Visualisation ─────────────────────────────────────────────────────────────
FIG_DPI    = 150
STYLE      = "seaborn-v0_8-whitegrid"
PALETTE    = "tab10"
