import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from analysis import cfr_by_country
from arima_model import run_arima_pipeline
from config import FORECAST_COUNTRY, FORECAST_HORIZON, TARGET_COUNTRIES
from etl_pipeline import load_clean_data, run_etl as run_full_etl
from visualizations import build_dashboard
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "covid_backend.db")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
INDEX_FILE = os.path.join(FRONTEND_DIR, "index.html")

app = FastAPI(title="COVID Analysis Backend", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_DB_LOCK = threading.Lock()
_RATE_LOCK = threading.Lock()
_WORKER: threading.Thread | None = None
_CANCELLED: set[str] = set()

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "480"))
RATE_LIMIT_PER_MIN = 120
RATE_BUCKET: dict[tuple[str, str], list[float]] = {}

USERS = {
    "admin": {"password": "covid123", "role": "admin"},
    "analyst": {"password": "analyst123", "role": "analyst"},
    "viewer": {"password": "viewer123", "role": "viewer"},
}
ROLE_RANK = {"viewer": 1, "analyst": 2, "admin": 3}


def run_etl():
    """Load cached clean data for API jobs, falling back to the full ETL."""
    try:
        return load_clean_data()
    except Exception:
        return run_full_etl()


class APIError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    expires_at: str


class RunRequest(BaseModel):
    country: str = Field(default=FORECAST_COUNTRY)
    horizon: int = Field(default=30, ge=7, le=90)
    with_dashboard: bool = False


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in row.keys():
        v = row[k]
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def init_db() -> None:
    with _DB_LOCK:
        with get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    country TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    total_records INTEGER,
                    countries_count INTEGER,
                    start_date TEXT,
                    end_date TEXT,
                    arima_order TEXT,
                    mape REAL,
                    rmse REAL,
                    mae REAL,
                    dashboard_path TEXT,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS top_cfr (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    rank_no INTEGER NOT NULL,
                    country TEXT NOT NULL,
                    cfr_pct REAL NOT NULL,
                    total_cases INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecast_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    forecast_date TEXT NOT NULL,
                    forecast_value REAL NOT NULL,
                    lower_bound REAL NOT NULL,
                    upper_bound REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS api_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    username TEXT,
                    role TEXT,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    ip_address TEXT,
                    duration_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pipeline_jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT,
                    payload_json TEXT NOT NULL,
                    run_id INTEGER,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS assistant_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT,
                    metadata_json TEXT,
                    content TEXT NOT NULL
                );
                """
            )
            conn.commit()


def _audit(request_id: str, method: str, path: str, status_code: int, duration_ms: int, ip: str | None, user: str | None, role: str | None) -> None:
    with _DB_LOCK:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO api_audit_logs (created_at, request_id, username, role, method, path, status_code, ip_address, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now_utc().isoformat(), request_id, user, role, method, path, status_code, ip, duration_ms),
            )
            conn.commit()


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    req_id = getattr(request.state, "request_id", secrets.token_hex(6))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details, "request_id": req_id}},
    )


def _apply_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    key = (ip, request.url.path)
    now = time.time()
    with _RATE_LOCK:
        RATE_BUCKET[key] = [t for t in RATE_BUCKET.get(key, []) if t >= now - 60]
        if len(RATE_BUCKET[key]) >= RATE_LIMIT_PER_MIN:
            raise APIError("RATE_LIMIT_EXCEEDED", "Too many requests.", status.HTTP_429_TOO_MANY_REQUESTS)
        RATE_BUCKET[key].append(now)


def _sign_token(payload: dict[str, Any]) -> str:
    import base64
    import hashlib
    import hmac
    header = {"alg": "HS256", "typ": "JWT"}
    hb = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode()).rstrip(b"=")
    pb = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    msg = hb + b"." + pb
    sig = hmac.new(JWT_SECRET.encode(), msg, hashlib.sha256).digest()
    sb = base64.urlsafe_b64encode(sig).rstrip(b"=")
    return (msg + b"." + sb).decode()


def _verify_token(token: str) -> dict[str, Any]:
    import base64
    import hashlib
    import hmac
    try:
        h, p, s = token.split(".")
    except ValueError as exc:
        raise APIError("AUTH_INVALID_TOKEN", "Invalid token format.", status.HTTP_401_UNAUTHORIZED) from exc
    msg = f"{h}.{p}".encode()
    exp_sig = base64.urlsafe_b64encode(hmac.new(JWT_SECRET.encode(), msg, hashlib.sha256).digest()).rstrip(b"=").decode()
    if not hmac.compare_digest(exp_sig, s):
        raise APIError("AUTH_INVALID_TOKEN", "Invalid token signature.", status.HTTP_401_UNAUTHORIZED)
    padded = p + "=" * ((4 - len(p) % 4) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
    if int(payload.get("exp", 0)) < int(_now_utc().timestamp()):
        raise APIError("AUTH_TOKEN_EXPIRED", "Token expired.", status.HTTP_401_UNAUTHORIZED)
    return payload


def _token_from_request(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise APIError("AUTH_MISSING", "Missing bearer token.", status.HTTP_401_UNAUTHORIZED)
    return auth[7:].strip()


def get_current_user(request: Request) -> dict[str, Any]:
    auth = request.headers.get("Authorization", "").strip()
    if not auth:
        return {"username": "guest", "role": "admin"}
    payload = _verify_token(_token_from_request(request))
    return {"username": payload["sub"], "role": payload["role"]}


def require_role(min_role: str):
    def dep(user: dict[str, Any] = Depends(get_current_user)):
        if ROLE_RANK.get(user["role"], 0) < ROLE_RANK.get(min_role, 0):
            raise APIError("AUTH_FORBIDDEN", f"Requires role: {min_role}", status.HTTP_403_FORBIDDEN)
        return user
    return dep


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request.state.request_id = secrets.token_hex(8)
    started = time.time()
    _apply_rate_limit(request)
    user = role = None
    try:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            p = _verify_token(auth[7:].strip())
            user, role = p.get("sub"), p.get("role")
    except Exception:
        pass
    response = await call_next(request)
    _audit(request.state.request_id, request.method, request.url.path, response.status_code, int((time.time() - started) * 1000), request.client.host if request.client else None, user, role)
    response.headers["X-Request-Id"] = request.state.request_id
    return response


def _job_update(job_id: str, **fields: Any) -> None:
    fields["updated_at"] = _now_utc().isoformat()
    set_sql = ", ".join(f"{k}=?" for k in fields.keys())
    vals = list(fields.values()) + [job_id]
    with _DB_LOCK:
        with get_conn() as conn:
            conn.execute(f"UPDATE pipeline_jobs SET {set_sql} WHERE id=?", vals)
            conn.commit()


def _job_pick_next() -> sqlite3.Row | None:
    with get_conn() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM pipeline_jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1").fetchone()
            if not row:
                conn.commit()
                return None

            updated = conn.execute(
                "UPDATE pipeline_jobs SET status='running', progress=5, message='Worker picked job', updated_at=? WHERE id=? AND status='queued'",
                (_now_utc().isoformat(), row["id"]),
            ).rowcount
            if updated != 1:
                conn.rollback()
                return None

            conn.commit()
            return conn.execute("SELECT * FROM pipeline_jobs WHERE id=?", (row["id"],)).fetchone()
        except sqlite3.OperationalError:
            conn.rollback()
            return None


def _job_cancelled(job_id: str) -> bool:
    if job_id in _CANCELLED:
        return True
    with _DB_LOCK:
        with get_conn() as conn:
            row = conn.execute("SELECT status FROM pipeline_jobs WHERE id=?", (job_id,)).fetchone()
    return bool(row and row["status"] == "cancelling")


def _process_job(job_id: str, payload: RunRequest) -> None:
    try:
        _job_update(job_id, progress=10, message="Creating run record")
        with _DB_LOCK:
            with get_conn() as conn:
                cur = conn.execute(
                    "INSERT INTO pipeline_runs (created_at, status, country, horizon) VALUES (?, 'running', ?, ?)",
                    (_now_utc().isoformat(), payload.country, payload.horizon),
                )
                run_id = cur.lastrowid
                conn.execute("UPDATE pipeline_jobs SET run_id=? WHERE id=?", (run_id, job_id))
                conn.commit()
        if _job_cancelled(job_id):
            raise RuntimeError("Cancelled by user")
        _job_update(job_id, progress=25, message="Running ETL")
        df = run_etl()
        _job_update(job_id, progress=45, message="Running analysis")
        cfr_df = cfr_by_country(df).head(10).reset_index(drop=True)
        if _job_cancelled(job_id):
            raise RuntimeError("Cancelled by user")
        _job_update(job_id, progress=62, message="Running forecasting")
        arima = run_arima_pipeline(df=df, country=payload.country, horizon=payload.horizon)
        dashboard_path = None
        if payload.with_dashboard:
            _job_update(job_id, progress=80, message="Generating dashboard")
            from analysis import monthly_global_trend, regional_burden
            dashboard_path = build_dashboard(df, cfr_df, regional_burden(df), monthly_global_trend(df), arima)
        m = arima["metrics"]
        with _DB_LOCK:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE pipeline_runs SET status='completed', total_records=?, countries_count=?, start_date=?, end_date=?, arima_order=?, mape=?, rmse=?, mae=?, dashboard_path=? WHERE id=?",
                    (int(len(df)), int(df["Country"].nunique()), str(df["Date_reported"].min().date()), str(df["Date_reported"].max().date()), f"ARIMA{tuple(arima['order'])}", float(m["MAPE"]), float(m["RMSE"]), float(m["MAE"]), dashboard_path, run_id),
                )
                conn.execute("DELETE FROM top_cfr WHERE run_id=?", (run_id,))
                for rank, (_, row) in enumerate(cfr_df.iterrows(), start=1):
                    conn.execute("INSERT INTO top_cfr (run_id, rank_no, country, cfr_pct, total_cases) VALUES (?, ?, ?, ?, ?)", (run_id, rank, str(row["Country"]), float(row["CFR_pct"]), int(row["Total_cases"])))
                conn.execute("DELETE FROM forecast_points WHERE run_id=?", (run_id,))
                for d, v, lo, hi in zip(arima["forecast_dates"], arima["forecast_values"], arima["forecast_lower"], arima["forecast_upper"]):
                    conn.execute("INSERT INTO forecast_points (run_id, forecast_date, forecast_value, lower_bound, upper_bound) VALUES (?, ?, ?, ?, ?)", (run_id, str(d.date()), float(v), float(lo), float(hi)))
                conn.commit()
        _job_update(job_id, status="completed", progress=100, message="Completed")
    except Exception as exc:  # noqa: BLE001
        status_txt = "cancelled" if "Cancelled by user" in str(exc) else "failed"
        with _DB_LOCK:
            with get_conn() as conn:
                row = conn.execute("SELECT run_id FROM pipeline_jobs WHERE id=?", (job_id,)).fetchone()
                if row and row["run_id"]:
                    conn.execute("UPDATE pipeline_runs SET status='failed', error=? WHERE id=?", (str(exc), row["run_id"]))
                    conn.commit()
        _job_update(job_id, status=status_txt, progress=100, message=str(exc), error=str(exc))
    finally:
        _CANCELLED.discard(job_id)


def _worker_loop() -> None:
    while True:
        row = _job_pick_next()
        if not row:
            time.sleep(0.5)
            continue
        payload = RunRequest(**json.loads(row["payload_json"]))
        _process_job(row["id"], payload)


def _ensure_worker() -> None:
    global _WORKER
    if _WORKER and _WORKER.is_alive():
        return
    _WORKER = threading.Thread(target=_worker_loop, daemon=True)
    _WORKER.start()


def _recover_jobs() -> None:
    with _DB_LOCK:
        with get_conn() as conn:
            conn.execute("UPDATE pipeline_jobs SET status='queued', message='Recovered after restart', updated_at=? WHERE status='running'", (_now_utc().isoformat(),))
            conn.commit()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    _recover_jobs()
    _ensure_worker()


@app.get("/", include_in_schema=False)
def frontend_home() -> FileResponse:
    if not os.path.exists(INDEX_FILE):
        raise APIError("FRONTEND_NOT_FOUND", "Frontend file not found.", status.HTTP_404_NOT_FOUND)
    return FileResponse(INDEX_FILE)


@app.get("/frontend/{asset_name}", include_in_schema=False)
def frontend_asset(asset_name: str) -> FileResponse:
    if asset_name not in {"styles.css", "app.js"}:
        raise APIError("ASSET_NOT_FOUND", "Asset not found.", status.HTTP_404_NOT_FOUND)
    p = os.path.join(FRONTEND_DIR, asset_name)
    if not os.path.exists(p):
        raise APIError("ASSET_NOT_FOUND", "Asset not found.", status.HTTP_404_NOT_FOUND)
    media = "text/css" if asset_name.endswith(".css") else "application/javascript"
    return FileResponse(p, media_type=media)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "database": DB_PATH, "worker": "running" if (_WORKER and _WORKER.is_alive()) else "stopped"}


@app.post("/auth/login", response_model=TokenResponse)
def auth_login(payload: TokenRequest) -> TokenResponse:
    username = payload.username.strip().lower()
    password = payload.password.strip()
    u = USERS.get(username)
    if not u or u["password"] != password:
        raise APIError("AUTH_INVALID_CREDENTIALS", "Invalid username or password.", status.HTTP_401_UNAUTHORIZED)
    issued = _now_utc()
    exp = issued + timedelta(minutes=JWT_EXPIRES_MINUTES)
    token = _sign_token({"sub": username, "role": u["role"], "iat": int(issued.timestamp()), "exp": int(exp.timestamp())})
    return TokenResponse(access_token=token, role=u["role"], username=username, expires_at=exp.isoformat())


@app.get("/auth/me")
def auth_me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return user


@app.get("/api/countries")
def list_countries(user: dict[str, Any] = Depends(require_role("viewer"))) -> dict[str, Any]:
    return {"countries": sorted(TARGET_COUNTRIES), "default_country": FORECAST_COUNTRY}


@app.get("/api/summary")
def get_summary(user: dict[str, Any] = Depends(require_role("viewer"))) -> dict[str, Any]:
    with get_conn() as conn:
        total_runs = int(conn.execute("SELECT COUNT(*) AS total FROM pipeline_runs").fetchone()["total"])
        running_jobs = int(conn.execute("SELECT COUNT(*) AS total FROM pipeline_jobs WHERE status='running'").fetchone()["total"])
        latest_run = conn.execute("SELECT id, created_at, status, country, horizon FROM pipeline_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    return {
        "tracked_countries": len(TARGET_COUNTRIES),
        "default_horizon": FORECAST_HORIZON,
        "total_runs": total_runs,
        "running_jobs": running_jobs,
        "live_analytics": "Active" if running_jobs > 0 else "Ready",
        "latest_run": _row_to_dict(latest_run) if latest_run else None,
    }


@app.post("/api/jobs")
def create_job(payload: RunRequest, user: dict[str, Any] = Depends(require_role("analyst"))) -> dict[str, Any]:
    if payload.country not in TARGET_COUNTRIES:
        raise APIError("INVALID_COUNTRY", "Selected country is not supported.", status.HTTP_400_BAD_REQUEST)
    job_id = secrets.token_hex(12)
    now = _now_utc().isoformat()
    with _DB_LOCK:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO pipeline_jobs (id, created_at, updated_at, created_by, status, progress, message, payload_json) VALUES (?, ?, ?, ?, 'queued', 0, 'Queued', ?)",
                (job_id, now, now, user["username"], json.dumps(payload.model_dump())),
            )
            conn.commit()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: dict[str, Any] = Depends(require_role("viewer"))) -> dict[str, Any]:
    with _DB_LOCK:
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM pipeline_jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise APIError("JOB_NOT_FOUND", f"Job id {job_id} not found.", status.HTTP_404_NOT_FOUND)
    job = _row_to_dict(row)
    if job.get("status") == "completed" and job.get("run_id"):
        job["result"] = {"run_id": job["run_id"]}
    return {"job": job}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: dict[str, Any] = Depends(require_role("analyst"))) -> dict[str, Any]:
    with _DB_LOCK:
        with get_conn() as conn:
            row = conn.execute("SELECT status FROM pipeline_jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise APIError("JOB_NOT_FOUND", f"Job id {job_id} not found.", status.HTTP_404_NOT_FOUND)
            if row["status"] in {"completed", "failed", "cancelled"}:
                return {"job_id": job_id, "status": row["status"], "message": "Job already finished."}
            conn.execute("UPDATE pipeline_jobs SET status='cancelling', message='Cancellation requested', updated_at=? WHERE id=?", (_now_utc().isoformat(), job_id))
            conn.commit()
    _CANCELLED.add(job_id)
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/api/runs")
def list_runs(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    country: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    user: dict[str, Any] = Depends(require_role("viewer")),
) -> dict[str, Any]:
    where, params = [], []
    if status:
        where.append("status=?"); params.append(status)
    if country:
        where.append("country=?"); params.append(country)
    if date_from:
        where.append("created_at>=?"); params.append(date_from)
    if date_to:
        where.append("created_at<=?"); params.append(date_to)
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    with get_conn() as conn:
        total = int(conn.execute(f"SELECT COUNT(*) AS total FROM pipeline_runs {where_sql}", tuple(params)).fetchone()["total"])
        rows = conn.execute(
            f"""SELECT id, created_at, status, country, horizon, total_records, countries_count, start_date, end_date, arima_order, mape, rmse, mae, dashboard_path, error
            FROM pipeline_runs {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?""",
            tuple(params + [limit, offset]),
        ).fetchall()
    return {"runs": [_row_to_dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@app.get("/api/runs/{run_id}")
def get_run(run_id: int, user: dict[str, Any] = Depends(require_role("viewer"))) -> dict[str, Any]:
    with get_conn() as conn:
        run = conn.execute(
            "SELECT id, created_at, status, country, horizon, total_records, countries_count, start_date, end_date, arima_order, mape, rmse, mae, dashboard_path, error FROM pipeline_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        if not run:
            raise APIError("RUN_NOT_FOUND", f"Run id {run_id} not found.", status.HTTP_404_NOT_FOUND)
        cfr = conn.execute("SELECT rank_no, country, cfr_pct, total_cases FROM top_cfr WHERE run_id=? ORDER BY rank_no ASC", (run_id,)).fetchall()
        fc = conn.execute("SELECT forecast_date, forecast_value, lower_bound, upper_bound FROM forecast_points WHERE run_id=? ORDER BY forecast_date ASC", (run_id,)).fetchall()
    return {"run": _row_to_dict(run), "top_cfr": [_row_to_dict(r) for r in cfr], "forecast": [_row_to_dict(r) for r in fc]}


@app.get("/api/runs/{run_id}/dashboard")
def get_dashboard_image(run_id: int, user: dict[str, Any] = Depends(require_role("viewer"))) -> FileResponse:
    with get_conn() as conn:
        run = conn.execute("SELECT dashboard_path FROM pipeline_runs WHERE id=?", (run_id,)).fetchone()
    if not run:
        raise APIError("RUN_NOT_FOUND", f"Run id {run_id} not found.", status.HTTP_404_NOT_FOUND)
    path = run["dashboard_path"]
    if not path:
        raise APIError("DASHBOARD_NOT_AVAILABLE", "No dashboard image stored for this run.", status.HTTP_404_NOT_FOUND)
    if not os.path.exists(path):
        raise APIError("DASHBOARD_FILE_MISSING", "Dashboard file path exists in DB but file is missing.", status.HTTP_404_NOT_FOUND)
    return FileResponse(path, media_type="image/png")


# ----------------------
# Lightweight RAG assistant
# ----------------------


class AssistantQuery(BaseModel):
    question: str
    top_k: int = 3


@app.post("/assistant/index")
def assistant_index(user: dict[str, Any] = Depends(require_role("analyst"))) -> dict[str, Any]:
    """Build a lightweight retrieval corpus from recent runs and summaries.
    This regenerates the `assistant_docs` table with one document per run.
    """
    with get_conn() as conn:
        runs = conn.execute("SELECT id, created_at, status, country, horizon, mape, rmse, mae FROM pipeline_runs ORDER BY id DESC LIMIT 200").fetchall()

    docs = []
    with _DB_LOCK:
        with get_conn() as conn:
            conn.execute("DELETE FROM assistant_docs")
            for r in runs:
                rid = r[0]
                # fetch top CFR and few forecast points
                with get_conn() as conn2:
                    cfr_rows = conn2.execute("SELECT rank_no, country, cfr_pct FROM top_cfr WHERE run_id=? ORDER BY rank_no ASC LIMIT 5", (rid,)).fetchall()
                    fc_rows = conn2.execute("SELECT forecast_date, forecast_value FROM forecast_points WHERE run_id=? ORDER BY forecast_date ASC LIMIT 5", (rid,)).fetchall()

                summary_parts = [f"Run {rid}: {r['status']} | {r['country']} | horizon={r['horizon']}d | MAPE={r['mape'] or 'NA'} | RMSE={r['rmse'] or 'NA'} | MAE={r['mae'] or 'NA'}"]
                if cfr_rows:
                    summary_parts.append("Top CFR: " + ", ".join([f"{c['country']}({c['cfr_pct']:.2f}%)" for c in cfr_rows]))
                if fc_rows:
                    sample = ", ".join([f"{f['forecast_date']}:{int(f['forecast_value'])}" for f in fc_rows])
                    summary_parts.append("Forecast sample: " + sample)

                content = " \n ".join(summary_parts)
                conn.execute("INSERT INTO assistant_docs (created_at, source, metadata_json, content) VALUES (?, ?, ?, ?)", (_now_utc().isoformat(), f"run:{rid}", json.dumps({"run_id": rid}), content))
            conn.commit()

    return {"indexed": True, "documents": len(runs)}


@app.post("/assistant/query")
def assistant_query(payload: AssistantQuery, user: dict[str, Any] = Depends(require_role("viewer"))) -> dict[str, Any]:
    """Retrieve top-K contexts and return a simple generated answer using them.
    This is a lightweight RAG: retrieval via TF-IDF + template-based generation.
    """
    q = payload.question.strip()
    if not q:
        raise APIError("NO_QUERY", "Question must not be empty.")

    with get_conn() as conn:
        rows = conn.execute("SELECT id, content, source, metadata_json FROM assistant_docs ORDER BY id DESC").fetchall()
    docs = [r[1] for r in rows]
    ids = [r[0] for r in rows]
    sources = [r[2] for r in rows]

    if not docs:
        return {"answer": "No indexed documents. Run /assistant/index first.", "contexts": []}

    # Vectorize and get cosine similarity
    try:
        vec = TfidfVectorizer(stop_words='english', max_df=0.9)
        X = vec.fit_transform(docs + [q])
        sims = cosine_similarity(X[-1], X[:-1]).flatten()
        ranked = sims.argsort()[::-1][: payload.top_k]
    except Exception:
        # fallback: simple substring match
        ranked = sorted(range(len(docs)), key=lambda i: (q.lower() in docs[i].lower()), reverse=True)[: payload.top_k]

    contexts = []
    for i in ranked:
        contexts.append({"id": ids[i], "source": sources[i], "content": docs[i]})

    # Simple template generation: stitch top contexts and answer heuristically
    answer_lines = [f"Retrieved {len(contexts)} context(s):"]
    for c in contexts:
        answer_lines.append(f"- {c['content']}")

    # Very small heuristic 'answer' extraction: look for numbers or keywords
    heuristic = "I found these summaries relevant to your question. Review the contexts above for details."
    answer = "\n\n".join([heuristic, "\n".join(answer_lines)])

    return {"question": q, "answer": answer, "contexts": contexts}
