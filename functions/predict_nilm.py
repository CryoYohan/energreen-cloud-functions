# predict_nilm.py
import os
import io
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

import numpy as np
import joblib
from google.cloud import firestore, storage

# ---- Configuration ----
PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET = os.environ.get("MODEL_BUCKET") or "energreen-prediction-model"

# Firestore & GCS clients
db = firestore.Client(project=PROJECT_ID)
gcs_client = storage.Client(project=PROJECT_ID)

# ---- FastAPI app ----
app = FastAPI(title="NILM Predict Appliance Service")

# ---- CORS configuration ----
FRONTEND_ORIGINS = [
    "http://localhost:5173",    # local dev
    "https://your-production-frontend.com",  # replace with real frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Model cache ----
_model_cache = {}

# ---- New Request and Response Schemas ----

class ApplianceSignature(BaseModel):
    # This is the unique ID of the Firestore document for the signature
    _id: str 
    signature: List[Dict[str, Any]]

class BatchPredictRequest(BaseModel):
    device_id: str
    signatures: List[ApplianceSignature]

class PredictionResult(BaseModel):
    _id: str
    predicted_label: str
    predicted_probabilities: Dict[str, float] | None = None
    
class BatchPredictResponse(BaseModel):
    device_id: str
    predictions: List[PredictionResult]
    model_used: str
    timestamp: str


# ---- Vectorization function ----
SAMPLE_LENGTH = 12
def signature_to_vector(sig: List[Dict[str, Any]], sample_len: int = SAMPLE_LENGTH):
    if not sig:
        return np.array([0.0]*sample_len + [0.0]*7, dtype=float)

    power, timestamps = [], []
    for rec in sig:
        pw = rec.get("powerWatt") or rec.get("power") or rec.get("power_watt") or 0.0
        try: pw = float(pw)
        except: pw = 0.0
        power.append(pw)
        ts = rec.get("timestamp", None)
        try: timestamps.append(float(ts))
        except: timestamps.append(None)

    series = (power[-sample_len:] if len(power) >= sample_len else [0.0]*(sample_len-len(power)) + power)
    arr = np.array(power if power else [0.0])
    mean, std, mn, mx, med = float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr)), float(np.median(arr))
    valid_ts = [t for t in timestamps if t is not None]
    duration = float(max(valid_ts)-min(valid_ts)) if len(valid_ts)>=2 else float(len(power))
    energy = (float(np.sum(arr)*(max(valid_ts)-min(valid_ts))/max(1,len(valid_ts)-1))/3600.0) if len(valid_ts)>=2 else float(np.sum(arr))/3600.0
    stats = [mean, std, mn, mx, med, duration, energy]
    return np.array(series + stats, dtype=float)

# ---- Root endpoint ----
@app.get("/")
def root():
    return {"message": "Prediction service is alive!"}

# ---- Load model helper ----
def load_latest_model(device_id: str):
    global _model_cache
    if device_id in _model_cache:
        return _model_cache[device_id]

    device_doc_ref = db.collection("devices").document(device_id)
    device_doc = device_doc_ref.get()
    if not device_doc.exists:
        raise HTTPException(status_code=404, detail="Device not found in Firestore")

    latest_model_info = device_doc.to_dict().get("latest_model")
    if not latest_model_info:
        raise HTTPException(status_code=404, detail="No trained model found for this device")

    model_gcs_uri = latest_model_info.get("model_gcs_uri")
    if not model_gcs_uri:
        raise HTTPException(status_code=404, detail="Model GCS URI not found")

    try:
        bucket_name = model_gcs_uri.split("/")[2]
        blob_path = "/".join(model_gcs_uri.split("/")[3:])
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        model_bytes = blob.download_as_bytes()
        clf = joblib.load(io.BytesIO(model_bytes))
        _model_cache[device_id] = clf
        logging.info(f"Loaded and cached model for device {device_id}")
        return clf
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model from GCS: {e}")

# ---- Predict endpoint ----
@app.post("/predict-appliance", response_model=BatchPredictResponse)
async def predict_appliance(req: BatchPredictRequest):
    if not req.device_id or not req.signatures:
        raise HTTPException(status_code=400, detail="device_id and signatures list are required")

    clf = load_latest_model(req.device_id)
    
    predictions = []
    
    # Initialize a Firestore batch
    batch = db.batch()
    
    for signature_data in req.signatures:
        X_vec = signature_to_vector(signature_data.signature).reshape(1, -1)
        
        try:
            pred_label = clf.predict(X_vec)[0]
            pred_proba = None
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_vec)[0]
                classes = clf.classes_
                pred_proba = {str(classes[i]): float(proba[i]) for i in range(len(classes))}
        except Exception as e:
            logging.error(f"Prediction failed for signature_id {signature_data._id}: {e}")
            pred_label = "unknown"
            pred_proba = None
        
        # Add update operation to the Firestore batch
        appliance_doc_ref = db.collection("devices").document(req.device_id).collection("appliance_predictions").document(signature_data._id)
        batch.update(appliance_doc_ref, {
            "predicted_label": pred_label,
            "confidence": pred_proba.get(pred_label, None) if pred_proba else None,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        
        # Append the prediction to the response list
        predictions.append(
            PredictionResult(
                _id=signature_data._id,
                predicted_label=pred_label,
                predicted_probabilities=pred_proba
            )
        )

    # Commit the Firestore batch
    batch.commit()
    
    return BatchPredictResponse(
        device_id=req.device_id,
        predictions=predictions,
        model_used=getattr(clf, "model_name", "unknown"),
        timestamp=datetime.now(timezone.utc).isoformat()
    )