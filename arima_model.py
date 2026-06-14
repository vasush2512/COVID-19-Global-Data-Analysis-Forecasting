"""
arima_model.py — ARIMA forecasting using pure NumPy + SciPy (no statsmodels)
"""
import logging, itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import FORECAST_COUNTRY, FORECAST_HORIZON, TRAIN_RATIO

logger = logging.getLogger(__name__)

# ── ADF Test ──────────────────────────────────────────────────────────────────
def _adf_test(series):
    y = np.array(series, dtype=float)
    dy = np.diff(y); y_l1 = y[:-1]
    X  = np.column_stack([y_l1, np.ones(len(y_l1))])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
        rss  = np.sum((dy - X @ coef)**2)
        denom = (len(dy)-2) * np.sum((y_l1 - y_l1.mean())**2)
        se   = np.sqrt(rss / denom) if denom > 0 else 1.0
        t    = coef[0] / se
        p    = float(stats.t.sf(abs(t), df=len(dy)-2) * 2)
    except Exception:
        t, p = 0.0, 1.0
    return {"adf_statistic": round(t,4), "p_value": round(p,4), "is_stationary": p < 0.05}

# ── Yule-Walker AR coefficients ───────────────────────────────────────────────
def _yule_walker(s, p):
    n = len(s); mu = s.mean(); sc = s - mu
    acf = [np.dot(sc[i:], sc[:n-i])/(n*np.var(sc)+1e-9) for i in range(p+1)]
    R   = np.array([[acf[abs(i-j)] for j in range(p)] for i in range(p)])
    r   = np.array(acf[1:p+1])
    try: return np.linalg.solve(R, r)
    except Exception: return np.zeros(p)

# ── Simple ARIMA(p,d,q) ───────────────────────────────────────────────────────
class SimpleARIMA:
    def __init__(self, p, d, q):
        self.p=p; self.d=d; self.q=q
        self.ar_coef=np.array([]); self.mu=0.; self.sigma=1.
        self._orig=None; self.aic=np.inf

    def fit(self, series):
        y = np.array(series, dtype=float); self._orig = y.copy()
        yd = y.copy()
        for _ in range(self.d): yd = np.diff(yd)
        self.mu = yd.mean(); ydc = yd - self.mu
        if self.p > 0 and len(ydc) > self.p:
            self.ar_coef = _yule_walker(ydc, self.p)
        res = self._resid(ydc)
        self.sigma = np.std(res) + 1e-9
        k  = self.p + self.q + 1
        ll = -len(res)/2*np.log(2*np.pi*self.sigma**2) - np.sum(res**2)/(2*self.sigma**2)
        self.aic = 2*k - 2*ll
        return self

    def _resid(self, ydc):
        n=len(ydc); res=np.zeros(n)
        for t in range(n):
            pred = sum(self.ar_coef[i]*ydc[t-i-1] for i in range(self.p) if t-i-1>=0)
            res[t] = ydc[t] - pred
        return res

    def forecast(self, steps):
        yd = self._orig.copy()
        for _ in range(self.d): yd = np.diff(yd)
        hist = list(yd - self.mu)
        preds = []
        for _ in range(steps):
            pred = self.mu + sum(self.ar_coef[i]*hist[-i-1] for i in range(self.p) if len(hist)>i)
            preds.append(pred); hist.append(pred - self.mu)
        preds = np.array(preds)
        # invert differencing
        if self.d > 0:
            base = self._orig[-1]; out = []
            running = base
            for p in preds: running += p; out.append(running)
            preds = np.array(out)
        return np.clip(preds, 0, None), self.sigma

# ── Grid Search ───────────────────────────────────────────────────────────────
def _grid_search(train, max_p=3, max_d=2, max_q=3):
    best_aic, best_order = np.inf, (1,1,1)
    for order in itertools.product(range(max_p+1), range(max_d+1), range(max_q+1)):
        try:
            m = SimpleARIMA(*order).fit(train)
            if m.aic < best_aic: best_aic=m.aic; best_order=order
        except: pass
    logger.info("Best ARIMA%s  AIC=%.2f", best_order, best_aic)
    return best_order

def _mape(a, p):
    mask = a != 0
    return float(np.mean(np.abs((a[mask]-p[mask])/a[mask]))*100)

def prepare_series(df, country):
    cdf = df[df["Country"]==country].sort_values("Date_reported").set_index("Date_reported")
    if cdf.empty: raise ValueError(f"Country '{country}' not found.")
    s = cdf["Cases_7day_avg"].dropna()
    return s[s > 0]

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_arima_pipeline(df=None, country=FORECAST_COUNTRY, horizon=FORECAST_HORIZON, train_ratio=TRAIN_RATIO):
    if df is None:
        from etl_pipeline import load_clean_data; df = load_clean_data()
    logger.info("═══ ARIMA | %s | horizon=%d ═══", country, horizon)
    series = prepare_series(df, country)
    vals   = series.values.astype(float)
    adf    = _adf_test(vals)
    logger.info("ADF p=%.4f stationary=%s", adf["p_value"], adf["is_stationary"])

    split = int(len(vals)*train_ratio)
    train_v, test_v = vals[:split], vals[split:]
    train_i, test_i = series.index[:split], series.index[split:]

    order = _grid_search(train_v)
    p,d,q = order

    # Walk-forward test predictions
    logger.info("Walk-forward test predictions (%d steps)…", len(test_v))
    history = list(train_v); test_preds = []
    for i in range(len(test_v)):
        fc,_ = SimpleARIMA(p,d,q).fit(np.array(history)).forecast(1)
        test_preds.append(fc[0]); history.append(test_v[i])
    test_preds = np.clip(np.array(test_preds), 0, None)

    metrics = {
        "MAPE": round(_mape(test_v, test_preds),2),
        "RMSE": round(float(np.sqrt(mean_squared_error(test_v, test_preds))),2),
        "MAE":  round(float(mean_absolute_error(test_v, test_preds)),2),
    }
    logger.info("MAPE=%.2f%%  RMSE=%.2f  MAE=%.2f", *metrics.values())

    # Full refit + forecast
    full_m = SimpleARIMA(p,d,q).fit(vals)
    fc_vals, sigma = full_m.forecast(horizon)
    margin  = 1.96 * sigma * np.sqrt(np.arange(1, horizon+1))
    fc_lower = np.clip(fc_vals - margin, 0, None)
    fc_upper = fc_vals + margin
    fc_dates = pd.date_range(start=series.index[-1]+pd.Timedelta(days=1), periods=horizon, freq="D")

    return {
        "country":          country,
        "order":            order,
        "adf":              adf,
        "train_series":     pd.Series(train_v, index=train_i),
        "test_series":      pd.Series(test_v,  index=test_i),
        "test_predictions": pd.Series(test_preds, index=test_i),
        "forecast_dates":   fc_dates,
        "forecast_values":  fc_vals,
        "forecast_lower":   fc_lower,
        "forecast_upper":   fc_upper,
        "metrics":          metrics,
        "fit_summary":      f"ARIMA({p},{d},{q}) | AIC={full_m.aic:.2f}",
    }

if __name__ == "__main__":
    from etl_pipeline import load_clean_data
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
    res = run_arima_pipeline(load_clean_data())
    m = res["metrics"]
    print(f"ARIMA{res['order']}  MAPE={m['MAPE']}%  RMSE={m['RMSE']:,.0f}  MAE={m['MAE']:,.0f}")
