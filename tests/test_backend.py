import sys
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backend  # noqa: E402


def _token(client: TestClient, username: str, password: str) -> str:
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_login_success_and_failure():
    with TestClient(backend.app) as client:
        ok = client.post("/auth/login", json={"username": "admin", "password": "covid123"})
        bad = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert ok.status_code == 200
        assert "access_token" in ok.json()
        assert bad.status_code == 401


def test_viewer_cannot_create_job():
    with TestClient(backend.app) as client:
        token = _token(client, "viewer", "viewer123")
        r = client.post(
            "/api/jobs",
            headers={"Authorization": f"Bearer {token}"},
            json={"country": "India", "horizon": 7, "with_dashboard": False},
        )
        assert r.status_code == 403


def test_job_lifecycle_fast_stubbed():
    original_run_etl = backend.run_etl
    original_cfr = backend.cfr_by_country
    original_arima = backend.run_arima_pipeline

    df = pd.DataFrame(
        {
            "Date_reported": pd.date_range("2021-01-01", periods=12, freq="D"),
            "Country": ["India"] * 12,
            "WHO_region": ["SEARO"] * 12,
            "New_cases": [10] * 12,
            "Cumulative_cases": list(range(10, 130, 10)),
            "New_deaths": [1] * 12,
            "Cumulative_deaths": list(range(1, 13)),
            "Cases_7day_avg": [10.0] * 12,
        }
    )

    def fake_etl():
        return df

    def fake_cfr(_):
        return pd.DataFrame([{"Country": "India", "CFR_pct": 1.0, "Total_cases": 120, "Total_deaths": 12, "WHO_region": "SEARO"}])

    def fake_arima(**_):
        dates = pd.date_range("2021-01-13", periods=7, freq="D")
        return {
            "order": (1, 1, 1),
            "metrics": {"MAPE": 1.0, "RMSE": 2.0, "MAE": 1.0},
            "forecast_dates": dates,
            "forecast_values": [10.0] * 7,
            "forecast_lower": [8.0] * 7,
            "forecast_upper": [12.0] * 7,
        }

    backend.run_etl = fake_etl
    backend.cfr_by_country = fake_cfr
    backend.run_arima_pipeline = fake_arima

    try:
        with TestClient(backend.app) as client:
            token = _token(client, "admin", "covid123")
            headers = {"Authorization": f"Bearer {token}"}
            create = client.post("/api/jobs", headers=headers, json={"country": "India", "horizon": 7, "with_dashboard": False})
            assert create.status_code == 200
            job_id = create.json()["job_id"]

            final = None
            for _ in range(60):
                j = client.get(f"/api/jobs/{job_id}", headers=headers)
                assert j.status_code == 200
                final = j.json()["job"]["status"]
                if final in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.1)
            assert final == "completed"
    finally:
        backend.run_etl = original_run_etl
        backend.cfr_by_country = original_cfr
        backend.run_arima_pipeline = original_arima
