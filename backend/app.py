import os
import time
import requests
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# ============================================================
# CONFIG
# ============================================================
load_dotenv(override=True)

SOBAT_API_URL = os.getenv("SOBAT_API_URL", "https://api.sobatberbagi.com/api/campaigns/")
SOBAT_API_KEY = os.getenv("SOBAT_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMPAIGN_BACKUP_JSON = os.getenv("CAMPAIGN_BACKUP_JSON")

if not CAMPAIGN_BACKUP_JSON:
    CAMPAIGN_BACKUP_JSON = os.path.join(BASE_DIR, "..", "data", "campaign_backup.json")
elif not os.path.isabs(CAMPAIGN_BACKUP_JSON):
    CAMPAIGN_BACKUP_JSON = os.path.join(BASE_DIR, CAMPAIGN_BACKUP_JSON)

CAMPAIGN_BACKUP_JSON = os.path.abspath(CAMPAIGN_BACKUP_JSON)
ONLINE_CSV = os.getenv("DONATIONS_ONLINE_CSV")
DONATIONS_XLSX = os.getenv("DONATIONS_XLSX", "data/jan-des 2025.xlsx") 
MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler_ma7.pkl")
META_PATH = os.getenv("META_PATH", "models/meta.pkl")

print("CWD:", os.getcwd())
print("MODEL_PATH:", MODEL_PATH)
print("DONATIONS_XLSX:", DONATIONS_XLSX)

if not SOBAT_API_KEY:
    raise RuntimeError("SOBAT_API_KEY belum diset. Isi di .env atau environment variable.")

# ============================================================
# LOAD MODEL + SCALER + META
# ============================================================
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal load model dari {MODEL_PATH}: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal load scaler dari {SCALER_PATH}: {e}")

try:
    meta = joblib.load(META_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal load meta dari {META_PATH}: {e}")

lookback = int(meta.get("lookback", 30))
feature_cols = meta.get("feature_cols", ["y", "dow", "dom", "month", "is_weekend"])
date_col = meta.get("date_col", "Tanggal")
value_col = meta.get("value_col", "pendapatan_offline_perhari")

# ============================================================
# CACHE BASELINE
# ============================================================
BASE_RAW_DF = None
BASE_FEAT_DF = None
BASE_SCALED_ALL = None

# cache hasil campaign dari SobatBerbagi biar tidak fetch terus
CAMPAIGN_CACHE = {
    "data": None,
    "fetched_at": 0.0,
}
CAMPAIGN_CACHE_TTL_SECONDS = 60

# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(
    title="Campaign + Prediction Backend",
    version="2.0",
    description="Backend FastAPI untuk mengambil list campaign dari SobatBerbagi dan menghitung prediksi hari tercapai."
)

# ============================================================
# REQUEST MODELS
# ============================================================
class CampaignPredictRequest(BaseModel):
    campaign_id: str = Field(..., description="ID campaign SobatBerbagi")
    max_days: int = Field(90, ge=1, le=365, description="Batas prediksi hari ke depan")
    preview_n: int = Field(14, ge=1, le=90, description="Jumlah hari preview yang dikembalikan")


# ============================================================
# HELPERS: SOBAT BERBAGI API
# ============================================================
def _extract_campaigns(payload):
    """
    Support beberapa format:
    - payload list langsung
    - payload dict: {"data": [ ... ]}
    - payload dict paginated: {"data": {"data": [ ... ]}}
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        d = payload.get("data")
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            dd = d.get("data")
            if isinstance(dd, list):
                return dd

    return None


def _safe_int(x):
    if x is None:
        return 0
    try:
        return int(float(str(x)))
    except Exception:
        return 0


def _safe_float(x):
    if x is None:
        return 0.0
    try:
        return float(str(x))
    except Exception:
        return 0.0

def save_campaigns_backup(campaigns: list[dict]):
    backup_dir = os.path.dirname(CAMPAIGN_BACKUP_JSON)

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)

    backup_payload = {
        "saved_at": datetime.now().isoformat(),
        "source_url": SOBAT_API_URL,
        "count": len(campaigns),
        "data": campaigns
    }

    with open(CAMPAIGN_BACKUP_JSON, "w", encoding="utf-8") as file:
        json.dump(
            backup_payload,
            file,
            ensure_ascii=False,
            indent=2
        )


def load_campaigns_backup() -> list[dict]:
    print(f"[INFO] Membaca backup dari: {CAMPAIGN_BACKUP_JSON}")

    if not os.path.exists(CAMPAIGN_BACKUP_JSON):
        raise HTTPException(
            status_code=503,
            detail=f"API SobatBerbagi tidak dapat diakses dan file backup campaign tidak ditemukan di path: {CAMPAIGN_BACKUP_JSON}"
        )

    try:
        with open(CAMPAIGN_BACKUP_JSON, "r", encoding="utf-8") as file:
            payload = json.load(file)

        campaigns = payload.get("data")

        if not isinstance(campaigns, list):
            raise ValueError("Format file backup tidak valid. Key 'data' bukan list.")

        print(f"[INFO] Backup campaign berhasil dibaca. Jumlah campaign: {len(campaigns)}")

        return campaigns

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membaca file backup campaign: {e}"
        )
    
def fetch_all_campaigns(force_refresh: bool = False) -> list[dict]:
    now = time.time()

    if (
        not force_refresh
        and CAMPAIGN_CACHE["data"] is not None
        and (now - CAMPAIGN_CACHE["fetched_at"]) < CAMPAIGN_CACHE_TTL_SECONDS
    ):
        return CAMPAIGN_CACHE["data"]

    headers = {"x-api-key": SOBAT_API_KEY}

    try:
        res = requests.get(SOBAT_API_URL, headers=headers, timeout=20)

        if res.status_code != 200:
            raise requests.RequestException(
                f"SobatBerbagi status {res.status_code}: {res.text[:300]}"
            )

        try:
            payload = res.json()
        except Exception as e:
            raise ValueError(f"Response SobatBerbagi bukan JSON valid: {e}")

        campaigns = _extract_campaigns(payload)

        if campaigns is None:
            raise ValueError("Format response SobatBerbagi tidak dikenali.")

        CAMPAIGN_CACHE["data"] = campaigns
        CAMPAIGN_CACHE["fetched_at"] = now

        # Kalau API berhasil, update backup terbaru
        save_campaigns_backup(campaigns)

        return campaigns

    except Exception as e:
        print(f"[WARNING] API SobatBerbagi gagal. Menggunakan backup offline. Error: {e}")

        campaigns = load_campaigns_backup()

        CAMPAIGN_CACHE["data"] = campaigns
        CAMPAIGN_CACHE["fetched_at"] = now

        return campaigns


def get_campaign_by_id(campaign_id: str) -> dict:
    campaigns = fetch_all_campaigns()
    campaign = next((c for c in campaigns if str(c.get("id")) == str(campaign_id)), None)
    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign_id tidak ditemukan di SobatBerbagi")
    return campaign


# ============================================================
# HELPERS: DATE / CALC
# ============================================================
def parse_date_safely(date_str):
    if not date_str:
        return None

    try:
        s = str(date_str).strip()

        # handle Z
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")

        dt = datetime.fromisoformat(s)
        return dt
    except Exception:
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            return None


def calc_days_left(end_date_str: str) -> int:
    dt = parse_date_safely(end_date_str)
    if dt is None:
        return 0

    today = datetime.now().date()
    end_date = dt.date()
    diff = (end_date - today).days
    return diff if diff > 0 else 0


# ============================================================
# LOAD DONATIONS FROM XLSX
# ============================================================
def load_donations_df() -> pd.DataFrame:
    # =========================
    # LOAD OFFLINE
    # =========================
    if not os.path.exists(DONATIONS_XLSX):
        raise FileNotFoundError(f"File offline tidak ditemukan: {DONATIONS_XLSX}")

    df_offline = pd.read_excel(DONATIONS_XLSX)
    df_offline[date_col] = pd.to_datetime(df_offline[date_col]).dt.date
    df_offline[value_col] = pd.to_numeric(
        df_offline[value_col], errors="coerce"
    ).fillna(0.0)

    # =========================
    # LOAD ONLINE
    # =========================
    if not os.path.exists(ONLINE_CSV):
        raise FileNotFoundError(f"File online tidak ditemukan: {ONLINE_CSV}")

    df_online = pd.read_csv(ONLINE_CSV)
    df_online["created_at"] = pd.to_datetime(df_online["created_at"])
    df_online["Tanggal"] = df_online["created_at"].dt.date

    online_daily = df_online.groupby("Tanggal")["amount"].sum().reset_index()
    online_daily.rename(columns={"amount": "pendapatan_online_perhari"}, inplace=True)

    # =========================
    # MERGE
    # =========================
    df = pd.merge(df_offline, online_daily, on="Tanggal", how="left")
    df["pendapatan_online_perhari"] = df["pendapatan_online_perhari"].fillna(0)

    # TOTAL
    df["total_donasi"] = (
        df["pendapatan_offline_perhari"] +
        df["pendapatan_online_perhari"]
    )

    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def build_scaled_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = raw_df.copy()

    df["ma7"] = df["total_donasi"].rolling(7).mean()
    df = df.dropna().reset_index(drop=True)

    df["dow"] = df[date_col].dt.dayofweek
    df["dom"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["y"] = np.log1p(df["ma7"].astype(float))

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature cols tidak ditemukan di df: {missing}")

    data = df[feature_cols].values.astype(float)
    scaled_all = scaler.transform(data)

    return df, scaled_all


def make_time_features_for_date(dt: pd.Timestamp) -> np.ndarray:
    dow = dt.dayofweek
    dom = dt.day
    month = dt.month
    is_weekend = 1 if dow >= 5 else 0
    return np.array([dow, dom, month, is_weekend], dtype=float)


def scale_row(y_log: float, time_feats: np.ndarray) -> np.ndarray:
    row = np.array([y_log, *time_feats], dtype=float).reshape(1, -1)
    return scaler.transform(row).flatten()


def inverse_scaled_y_to_rupiah_single(y_scaled: float) -> float:
    dummy = np.zeros((1, len(feature_cols)), dtype=float)
    dummy[0, 0] = float(y_scaled)

    inv = scaler.inverse_transform(dummy)
    y_log = float(inv[0, 0])

    y_rp = float(np.expm1(y_log))
    return max(0.0, y_rp)


# ============================================================
# CORE PREDICTION
# ============================================================
def predict_days_to_target(
    target_amount: int,
    feat_df: pd.DataFrame,
    scaled_all: np.ndarray,
    max_days: int = 90,
) -> tuple[int | None, list[str], list[float], list[float]]:
    if len(scaled_all) < lookback:
        raise ValueError(f"Data kurang setelah MA7. Minimal {lookback} baris, sekarang {len(scaled_all)}.")
    last_data_date = pd.to_datetime(feat_df[date_col].iloc[-1])
    # today = pd.Timestamp.today().normalize()
    # last_date = today if today > last_data_date else last_data_date // 1 januari
    # window = scaled_all[-lookback:, :].copy()
    today = pd.Timestamp.today().normalize()
    last_date = today 

    # print("[PREDIKSI] Tanggal hari ini:", today.date())
    # print("[PREDIKSI] Prediksi dimulai dari:", (last_date + pd.Timedelta(days=1)).date())
    window = scaled_all[-lookback:, :].copy() 
    # hari ini



    cum = 0.0
    dates_out: list[str] = []
    daily_preds: list[float] = []
    cum_preds: list[float] = []

    for day in range(1, max_days + 1):
        next_date = last_date + pd.Timedelta(days=day)

        X_in = window.reshape(1, lookback, window.shape[1])
        next_y_scaled = float(model.predict(X_in, verbose=0)[0, 0])

        next_value = inverse_scaled_y_to_rupiah_single(next_y_scaled)

        dates_out.append(next_date.date().isoformat())
        daily_preds.append(next_value)

        cum += next_value
        cum_preds.append(cum)

        if cum >= target_amount:
            return day, dates_out, daily_preds, cum_preds

        dummy = np.zeros((1, len(feature_cols)), dtype=float)
        dummy[0, 0] = next_y_scaled
        inv = scaler.inverse_transform(dummy)
        next_y_log = float(inv[0, 0])

        tfeat = make_time_features_for_date(next_date)
        next_row_scaled = scale_row(next_y_log, tfeat)

        window = np.vstack([window[1:, :], next_row_scaled.reshape(1, -1)])

    return None, dates_out, daily_preds, cum_preds


def predict_summary_for_campaign(campaign: dict, max_days: int = 90, preview_n: int = 14) -> dict:
    campaign_id = str(campaign.get("id"))
    title = campaign.get("title")
    target = _safe_int(campaign.get("target"))
    raised = _safe_int(campaign.get("raised"))
    remaining = max(0, target - raised)
    end_date = campaign.get("end_date")
    days_left = calc_days_left(end_date)

    result = {
        "campaign_id": campaign_id,

        "title": title,
        "thumbnail": campaign.get("thumbnail"),
        "target": target,
        "raised": raised,
        "remaining": remaining,
        "created_at": campaign.get("created_at"),
        "end_date": end_date,
        "days_left": days_left,
        "status": campaign.get("status"),
        "category": (
            campaign.get("campaign_category", {}).get("name")
            if isinstance(campaign.get("campaign_category"), dict)
            else None
        ),
        "estimated_days": None,
        "prediction_start_from": "tomorrow",
        "dates": [],
        "daily_forecast": [],
        "cumulative": [],
        "message": None,
    }

    if remaining <= 0:
        result["estimated_days"] = 0
        result["message"] = "Target sudah terpenuhi"
        return result

    try:
        feat_df = BASE_FEAT_DF
        scaled_all = BASE_SCALED_ALL

        days, dates_out, daily_preds, cum_preds = predict_days_to_target(
            target_amount=remaining,
            feat_df=feat_df,
            scaled_all=scaled_all,
            max_days=max_days
        )

        n = min(preview_n, len(daily_preds))

        result["estimated_days"] = days
        result["dates"] = dates_out[:n]
        result["daily_forecast"] = daily_preds[:n]
        result["cumulative"] = cum_preds[:n]

        if days is None:
            result["message"] = f"Target belum diperkirakan tercapai dalam {max_days} hari"
        else:
            result["message"] = f"Estimasi target tercapai dalam {days} hari"

        return result

    except Exception as e:
        result["message"] = f"Gagal prediksi: {e}"
        return result


# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
def startup_event():
    global BASE_RAW_DF, BASE_FEAT_DF, BASE_SCALED_ALL

    t0 = time.time()
    print("[STARTUP] Loading baseline XLSX...")

    BASE_RAW_DF = load_donations_df()
    BASE_FEAT_DF, BASE_SCALED_ALL = build_scaled_features(BASE_RAW_DF)

    t1 = time.time()
    print(f"[STARTUP] Baseline ready in {t1 - t0:.2f}s")
    print(f"[STARTUP] Rows raw: {len(BASE_RAW_DF)}, rows feat: {len(BASE_FEAT_DF)}")


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "message": "Campaign Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "lookback": lookback,
        "feature_cols": feature_cols,
        "baseline_xlsx": DONATIONS_XLSX,
        "sobat_api_url": SOBAT_API_URL,
        "baseline_cached": BASE_FEAT_DF is not None and BASE_SCALED_ALL is not None,
        "campaign_cache_ttl_seconds": CAMPAIGN_CACHE_TTL_SECONDS,
        "campaign_backup_path": CAMPAIGN_BACKUP_JSON,
        "campaign_backup_exists": os.path.exists(CAMPAIGN_BACKUP_JSON),
    }


@app.get("/campaigns/raw")
def get_raw_campaigns(refresh: bool = Query(False)):
    campaigns = fetch_all_campaigns(force_refresh=refresh)
    return {
        "count": len(campaigns),
        "data": campaigns,
    }


@app.get("/campaigns")
def get_campaigns(
    max_days: int = Query(90, ge=1, le=365),
    preview_n: int = Query(14, ge=1, le=90),
    refresh: bool = Query(False),
    limit: int = Query(0, ge=0),
):
    """
    Endpoint utama untuk Flutter.
    Ambil semua campaign + hasil prediksi.
    """
    t0 = time.time()

    campaigns = fetch_all_campaigns(force_refresh=refresh)

    if limit > 0:
        campaigns = campaigns[:limit]

    results = []
    for c in campaigns:
        results.append(
            predict_summary_for_campaign(
                campaign=c,
                max_days=max_days,
                preview_n=preview_n,
            )
        )

    t1 = time.time()

    return {
        "count": len(results),
        "max_days": max_days,
        "preview_n": preview_n,
        "processed_in_seconds": round(t1 - t0, 3),
        "data": results,
    }


@app.get("/campaigns/{campaign_id}")
def get_campaign_detail(
    campaign_id: str,
    max_days: int = Query(90, ge=1, le=365),
    preview_n: int = Query(14, ge=1, le=90),
    refresh: bool = Query(False),
):
    """
    Detail 1 campaign + prediksi.
    """
    if refresh:
        fetch_all_campaigns(force_refresh=True)

    campaign = get_campaign_by_id(campaign_id)
    result = predict_summary_for_campaign(
        campaign=campaign,
        max_days=max_days,
        preview_n=preview_n,
    )
    return result


@app.get("/debug-campaign/{campaign_id}")
def debug_campaign(campaign_id: str, refresh: bool = Query(False)):
    campaigns = fetch_all_campaigns(force_refresh=refresh)
    campaign = next((c for c in campaigns if str(c.get("id")) == str(campaign_id)), None)

    if campaign is None:
        raise HTTPException(status_code=404, detail="campaign_id tidak ditemukan")

    target = _safe_int(campaign.get("target"))
    raised = _safe_int(campaign.get("raised"))

    return {
        "campaign_id": campaign_id,
        "title": campaign.get("title"),
        "raw_target": campaign.get("target"),
        "raw_raised": campaign.get("raised"),
        "target": target,
        "raised": raised,
        "remaining": max(0, target - raised),
        "end_date": campaign.get("end_date"),
        "days_left": calc_days_left(campaign.get("end_date")),
        "status": campaign.get("status"),
        "updated_at": campaign.get("updated_at"),
    }


@app.post("/predict-campaign-days")
def predict_campaign_days(req: CampaignPredictRequest):
    """
    Kompatibel dengan endpoint lama.
    """
    campaign = get_campaign_by_id(req.campaign_id)
    result = predict_summary_for_campaign(
        campaign=campaign,
        max_days=req.max_days,
        preview_n=req.preview_n,
    )
    return result 