import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================
load_dotenv(override=True)

SOBAT_API_URL = os.getenv(
    "SOBAT_API_URL",
    "https://api.sobatberbagi.com/api/campaigns/"
)

SOBAT_API_KEY = os.getenv("SOBAT_API_KEY")

BACKUP_PATH = os.getenv(
    "CAMPAIGN_BACKUP_JSON",
    "data/campaigns_backup.json"
)

if not SOBAT_API_KEY:
    raise RuntimeError(
        "SOBAT_API_KEY belum diset. Isi dulu di file .env"
    )


# ============================================================
# HELPER EXTRACT RESPONSE
# ============================================================
def extract_campaigns(payload):
    """
    Mendukung beberapa kemungkinan format response:
    1. payload berupa list langsung
    2. payload berupa dict {"data": [...]}
    3. payload berupa dict pagination {"data": {"data": [...]}}
    """

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        data = payload.get("data")

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            nested_data = data.get("data")

            if isinstance(nested_data, list):
                return nested_data

    return None


# ============================================================
# FETCH DATA FROM API SOBAT BERBAGI
# ============================================================
def fetch_campaigns():
    headers = {
        "x-api-key": SOBAT_API_KEY
    }

    print("[INFO] Mengambil data campaign dari API Sobat Berbagi...")
    print(f"[INFO] URL: {SOBAT_API_URL}")

    try:
        response = requests.get(
            SOBAT_API_URL,
            headers=headers,
            timeout=30
        )
    except requests.RequestException as e:
        raise Exception(f"Gagal request ke API Sobat Berbagi: {e}")

    if response.status_code != 200:
        raise Exception(
            f"Gagal mengambil data API. "
            f"Status: {response.status_code}. "
            f"Response: {response.text[:300]}"
        )

    try:
        payload = response.json()
    except Exception as e:
        raise Exception(f"Response API bukan JSON valid: {e}")

    campaigns = extract_campaigns(payload)

    if campaigns is None:
        raise Exception(
            "Format response API tidak dikenali. "
            "Gagal mengambil list campaign."
        )

    return campaigns


# ============================================================
# SAVE BACKUP TO JSON
# ============================================================
def save_backup(campaigns):
    backup_dir = os.path.dirname(BACKUP_PATH)

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)

    backup_payload = {
        "saved_at": datetime.now().isoformat(),
        "source_url": SOBAT_API_URL,
        "count": len(campaigns),
        "data": campaigns
    }

    with open(BACKUP_PATH, "w", encoding="utf-8") as file:
        json.dump(
            backup_payload,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("[SUCCESS] Backup campaign berhasil dibuat.")
    print(f"[SUCCESS] Lokasi file: {BACKUP_PATH}")
    print(f"[SUCCESS] Jumlah campaign: {len(campaigns)}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        campaigns = fetch_campaigns()
        save_backup(campaigns)

    except Exception as e:
        print("[ERROR] Gagal membuat backup campaign.")
        print(f"[ERROR] Detail: {e}")