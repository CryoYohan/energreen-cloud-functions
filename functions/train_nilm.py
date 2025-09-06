# app.py
# Cloud Run service: Train NILM single-label model from Firestore appliance signatures.
# Expects POST /train-model with JSON: { "user_id": "...", "device_id": "...", "app_id": "optional-app-id" }
# Requires GOOGLE_APPLICATION_CREDENTIALS environment variable to be set (service account JSON)
# or default application credentials available to Cloud Run service.

import os
import time
import io
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from google.cloud import firestore
from google.cloud import storage

# ---- Configuration (via env vars) ----
PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET = os.environ.get("MODEL_BUCKET")          # e.g. "my-nilm-models-bucket"
MIN_SAMPLES_PER_CLASS = int(os.environ.get("MIN_SAMPLES_PER_CLASS", "5"))
SAMPLE_LENGTH = int(os.environ.get("SIGNATURE_SAMPLE_LEN", "12"))  # number of samples to use per signature

if not GCS_BUCKET:
    logging.warning("MODEL_BUCKET env var not set — model upload will fail unless provided.")

# Firestore & GCS clients (use ADC)
db = firestore.Client(project=PROJECT_ID)
gcs_client = storage.Client(project=PROJECT_ID)

app = FastAPI(title="NILM Train Model Service")

# ---- Request schema ----
class TrainRequest(BaseModel):
    user_id: str
    device_id: str
    app_id: Optional[str] = "default-app-id"
    force: Optional[bool] = False  # if True, trains even if low samples (not recommended)


# ---- Helper functions for feature extraction ----
def signature_to_vector(sig: List[Dict[str, Any]], sample_len: int = SAMPLE_LENGTH):
    """
    Convert a signature (list of readings dicts) into a fixed-length numeric vector.
    Strategy:
      - Extract powerWatt series (float)
      - Use last `sample_len` samples (pad with zeros on left if shorter)
      - Also compute summary stats: mean, std, min, max, median, duration, energy (approx)
    Returns concatenated vector (sample_len + summary_stats)
    """
    if not sig:
        # return zeros
        series = [0.0] * sample_len
        stats = [0.0] * 6
        return np.array(series + stats, dtype=float)

    # extract power series and timestamps (safe defaults)
    power = []
    timestamps = []
    for rec in sig:
        pw = rec.get("powerWatt") or rec.get("power") or rec.get("power_watt") or 0.0
        try:
            pw = float(pw)
        except Exception:
            pw = 0.0
        power.append(pw)
        ts = rec.get("timestamp", None)
        try:
            timestamps.append(float(ts))
        except Exception:
            timestamps.append(None)

    # convert to numpy
    power = [float(x) for x in power]

    # choose last sample_len values
    if len(power) >= sample_len:
        series = power[-sample_len:]
    else:
        # pad left with zeros so most recent samples are right-aligned
        pad = [0.0] * (sample_len - len(power))
        series = pad + power

    # summary stats
    arr = np.array(power) if len(power) > 0 else np.array([0.0])
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    med = float(np.median(arr))

    # duration: approximate difference between last and first timestamp (if available)
    valid_ts = [t for t in timestamps if t is not None]
    if len(valid_ts) >= 2:
        duration = float(max(valid_ts) - min(valid_ts))
    else:
        duration = float(len(power))  # fallback to count of samples

    # approximate energy: sum(power) * dt (dt approximated as 1 second per sample if timestamp missing)
    energy = 0.0
    if len(valid_ts) >= 2:
        # compute average dt
        dt = (max(valid_ts) - min(valid_ts)) / max(1, len(valid_ts)-1)
        energy = float(np.sum(arr) * dt) / 3600.0  # Wh -> kWh if arr in W and dt in seconds
    else:
        energy = float(np.sum(arr)) / 3600.0  # rough

    stats = [mean, std, mn, mx, med, duration, energy]
    # Return series + stats (series length + 7 stats)
    return np.array(series + stats, dtype=float)


def build_dataset_from_firestore(device_id: str, min_samples_per_class=MIN_SAMPLES_PER_CLASS):
    """
    Query Firestore for confirmed appliance_predictions for device_id, convert to dataset (X, y).
    Expects documents under: devices/{device_id}/appliance_predictions
    Only uses docs with status == 'confirmed' and confirmed_label set.
    """
    coll = db.collection("devices").document(device_id).collection("appliance_predictions")
    # Only confirmed labeled docs
    docs = list(coll.where("status", "==", "confirmed").stream())
    if not docs:
        return None, None, "No confirmed appliance predictions found for device."

    X = []
    y = []
    meta = []
    for d in docs:
        data = d.to_dict()
        label = data.get("confirmed_label") or data.get("label")
        signature = data.get("signature") or data.get("signature_data") or []
        if not label or not signature:
            continue
        vec = signature_to_vector(signature, SAMPLE_LENGTH)
        X.append(vec)
        y.append(label)
        meta.append({"doc_id": d.id, "raw": data})

    if len(X) == 0:
        return None, None, "No usable signature data found."

    # check class balance
    df = pd.DataFrame({"label": y})
    counts = df['label'].value_counts().to_dict()
    insufficient = {k: v for k, v in counts.items() if v < min_samples_per_class}
    if insufficient:
        return np.array(X), np.array(y), {
            "warning": "Some classes have few samples",
            "counts": counts
        }

    return np.array(X), np.array(y), {"counts": counts}


# ---- Endpoint: Train Model ----
@app.post("/train-model")
async def train_model(req: TrainRequest):
    start_ts = time.time()
    # Validate
    if not req.user_id or not req.device_id:
        raise HTTPException(status_code=400, detail="user_id and device_id are required")

    device_id = req.device_id
    user_id = req.user_id
    app_id = req.app_id or "default-app-id"

    # 1) Build dataset
    dataset = build_dataset_from_firestore(device_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="No confirmed appliance signatures found for this device.")

    X, y, info = dataset
    # If info is a dict warning, and force flag is False, reject
    if isinstance(info, dict) and info.get("warning") and not req.force:
        return {
            "status": "failed",
            "reason": "insufficient_samples",
            "message": "Some classes do not meet minimum sample count. Use force=true to override.",
            "details": info
        }

    # Convert X,y to numeric/dtype
    try:
        X = np.array(X, dtype=float)
        y = np.array(y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to construct numpy arrays: {e}")

    # Ensure at least 2 classes
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return {
            "status": "failed",
            "reason": "not_enough_classes",
            "message": "Need at least two different appliance labels to train a classifier."
        }

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3) Train a lightweight model (RandomForest)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)

    # 4) Evaluate
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 5) Save model to GCS
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = f"rf_model_{timestamp}.pkl"
    gcs_path = f"models/{user_id}/{device_id}/{model_name}"

    # Dump model to bytes via joblib
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    buf.seek(0)

    if not GCS_BUCKET:
        raise HTTPException(status_code=500, detail="GCS bucket not configured (set MODEL_BUCKET env var).")

    try:
        bucket = gcs_client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_file(buf, content_type="application/octet-stream")
        model_gcs_uri = f"gs://{GCS_BUCKET}/{gcs_path}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload model to GCS: {e}")

    # 6) Write model metadata to Firestore
    models_coll = db.collection("devices").document(device_id).collection("models")
    model_doc = {
        "user_id": user_id,
        "device_id": device_id,
        "app_id": app_id,
        "model_name": model_name,
        "model_gcs_uri": model_gcs_uri,
        "created_at": datetime.now(timezone.utc),
        "metrics": {
            "accuracy": acc,
            "report": report
        },
        "training_sample_count": int(len(y)),
        "labels": list(map(str, unique_labels))
    }
    try:
        model_ref = models_coll.document()
        model_ref.set(model_doc)
        model_id = model_ref.id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write model metadata to Firestore: {e}")

    # 7) Optionally annotate appliance_predictions with latest_model reference (not required)
    try:
        predictions_coll = db.collection("devices").document(device_id).collection("appliance_predictions")
        # write model reference into device top-level doc for quick lookup
        device_doc_ref = db.collection("devices").document(device_id)
        device_doc_ref.set({
            "latest_model": {
                "model_id": model_id,
                "model_name": model_name,
                "model_gcs_uri": model_gcs_uri,
                "created_at": datetime.now(timezone.utc)
            }
        }, merge=True)
    except Exception:
        # Not critical, log but don't fail the endpoint
        logging.exception("Warning: failed to update device latest_model metadata.")

    duration = time.time() - start_ts
    return {
        "status": "success",
        "model_id": model_id,
        "model_name": model_name,
        "model_gcs_uri": model_gcs_uri,
        "metrics": {
            "accuracy": acc,
            "report": report
        },
        "training_time_seconds": duration
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
