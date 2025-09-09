# train_nilm.py
# Cloud Run service: Train NILM single-label model from Firestore appliance signatures.

import os
import time
import io
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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


# ---- Configuration ----
PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET = "energreen-prediction-model"
MIN_SAMPLES_PER_CLASS = int(os.environ.get("MIN_SAMPLES_PER_CLASS", "5"))
SAMPLE_LENGTH = int(os.environ.get("SIGNATURE_SAMPLE_LEN", "12"))

db = firestore.Client(project=PROJECT_ID)
gcs_client = storage.Client(project=PROJECT_ID)

app = FastAPI(title="NILM Train Model Service")

# ---- CORS ----
FRONTEND_ORIGINS = [
    "http://localhost:5173",    # local dev
    "https://your-production-frontend.com",  # replace with real frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Training service is alive!"}


# ---- Request schema ----
class TrainRequest(BaseModel):
    user_id: str
    device_id: str
    app_id: Optional[str] = "default-app-id"
    force: Optional[bool] = False


# ---- Feature extraction ----
def signature_to_vector(sig: List[Dict[str, Any]], sample_len: int = SAMPLE_LENGTH):
    if not sig:
        return np.zeros(sample_len + 7, dtype=float)

    power = []
    timestamps = []
    for rec in sig:
        try:
            pw = float(rec.get("powerWatt") or rec.get("power") or rec.get("power_watt") or 0.0)
        except Exception:
            pw = 0.0
        power.append(pw)

        ts = rec.get("timestamp")
        try:
            timestamps.append(float(ts))
        except Exception:
            timestamps.append(None)

    # Pad or truncate power series
    if len(power) >= sample_len:
        series = power[-sample_len:]
    else:
        series = [0.0] * (sample_len - len(power)) + power

    arr = np.array(power) if power else np.array([0.0])
    mean, std, mn, mx, med = (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.median(arr)),
    )

    valid_ts = [t for t in timestamps if t is not None]
    if len(valid_ts) >= 2:
        duration = float(max(valid_ts) - min(valid_ts))
        dt = (max(valid_ts) - min(valid_ts)) / max(1, len(valid_ts) - 1)
        energy = float(np.sum(arr) * dt) / 3600.0
    else:
        duration = float(len(power))
        energy = float(np.sum(arr)) / 3600.0

    stats = [mean, std, mn, mx, med, duration, energy]
    return np.array(series + stats, dtype=float)


# ---- Dataset builder ----
def build_dataset_from_firestore(device_id: str, min_samples_per_class=MIN_SAMPLES_PER_CLASS):
    confirmed_coll = db.collection("devices").document(device_id).collection("confirmed_appliances")
    confirmed_docs = list(confirmed_coll.stream())

    X, y = [], []

    logging.info(f"Found {len(confirmed_docs)} confirmed_appliances for device {device_id}")

    for d in confirmed_docs:
        data = d.to_dict()

        # ✅ user_label is at the root level
        label = data.get("user_label")
        if not label or str(label).lower() in ["unknown", "unidentified", "none"]:
            continue

        # ✅ signatures are inside summary map
        summary = data.get("summary", {})
        signature_ids = summary.get("signatures", [])

        if not isinstance(signature_ids, list) or not label:
            continue


        logging.info(f"Confirmed appliance with label '{label}' has {len(signature_ids)} signatures")

        for sig_id in signature_ids:
            try:
                # fetch actual signature from appliance_predictions
                pred_ref = (
                    db.collection("devices")
                    .document(device_id)
                    .collection("appliance_predictions")
                    .document(sig_id)
                )
                pred_doc = pred_ref.get()
                if not pred_doc.exists:
                    logging.warning(f"Signature doc {sig_id} not found in appliance_predictions")
                    continue

                pred_data = pred_doc.to_dict()
                sig = pred_data.get("signature")
                if not sig or not isinstance(sig, list):
                    logging.warning(f"Signature doc {sig_id} missing 'signature' array")
                    continue

                vec = signature_to_vector(sig, SAMPLE_LENGTH)
                X.append(vec)
                y.append(label)

            except Exception as e:
                logging.warning(f"Skipping malformed signature {sig_id}: {e}")
                continue

    # Convert to arrays
    X, y = np.array(X), np.array(y)
    counts = pd.Series(y).value_counts().to_dict() if len(y) else {}

    logging.info(f"Built dataset with {len(X)} samples across {len(set(y))} classes: {counts}")

    # Filter out classes with too few samples
    classes_to_keep = [lbl for lbl, cnt in counts.items() if cnt >= min_samples_per_class]
    mask = np.isin(y, classes_to_keep)
    X, y = X[mask], y[mask]

    return X, y, {"counts": counts}

# ---- Train Model ----
@app.post("/train-model")
async def train_model(req: TrainRequest):
    start_ts = time.time()

    if not req.user_id or not req.device_id:
        raise HTTPException(status_code=400, detail="user_id and device_id are required")

    X, y, info = build_dataset_from_firestore(req.device_id)

    if len(y) == 0:
        return {
            "status": "failed",
            "reason": "no_data",
            "message": "No confirmed appliance signatures found for this device."
        }

    unique_labels = np.unique(y)
    if len(unique_labels) < 2 and not req.force:
        return {
            "status": "warning",
            "reason": "single_class",
            "message": "Only one device class available, training with UNKNOWN fallback.",
            "labels": list(map(str, unique_labels))
        }

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y if len(unique_labels) > 1 else None, random_state=42
        )

        clf = RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred)) if len(y_test) > 0 else 1.0
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Save model
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_name = f"rf_model_{timestamp}.pkl"
        gcs_path = f"models/{req.user_id}/{req.device_id}/{model_name}"

        buf = io.BytesIO()
        joblib.dump(clf, buf)
        buf.seek(0)

        bucket = gcs_client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.upload_from_file(buf, content_type="application/octet-stream")
        model_gcs_uri = f"gs://{GCS_BUCKET}/{gcs_path}"

        # Write Firestore metadata
        models_coll = db.collection("devices").document(req.device_id).collection("models")
        model_doc = {
            "user_id": req.user_id,
            "device_id": req.device_id,
            "app_id": req.app_id,
            "model_name": model_name,
            "model_gcs_uri": model_gcs_uri,
            "created_at": datetime.now(timezone.utc),
            "metrics": {"accuracy": acc, "report": report},
            "training_sample_count": int(len(y)),
            "labels": list(map(str, unique_labels)),
        }
        model_ref = models_coll.document()
        model_ref.set(model_doc)
        model_id = model_ref.id

        # Update latest_model
        db.collection("devices").document(req.device_id).set(
            {"latest_model": {
                "model_id": model_id,
                "model_name": model_name,
                "model_gcs_uri": model_gcs_uri,
                "created_at": datetime.now(timezone.utc)
            }},
            merge=True
        )

        return {
            "status": "success",
            "model_id": model_id,
            "model_name": model_name,
            "model_gcs_uri": model_gcs_uri,
            "metrics": {"accuracy": acc, "report": report},
            "training_time_seconds": time.time() - start_ts
        }

    except Exception as e:
        logging.exception("Training error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
