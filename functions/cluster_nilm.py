# cluster_nilm.py
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from google.cloud import firestore

# ---- Config ----
PROJECT_ID = os.getenv("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
FIRESTORE_CLIENT = firestore.Client(project=PROJECT_ID)

app = FastAPI()

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://your-production-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request Schema ----
class ClusterRequest(BaseModel):
    device_id: str
    n_clusters: int = 3


# ---- Endpoint ----
@app.post("/cluster-appliances")
async def cluster_appliances(req: ClusterRequest):
    device_id = req.device_id
    n_clusters = req.n_clusters

    try:
        # 1. Fetch signatures with 'unidentified' status
        predictions_ref = FIRESTORE_CLIENT.collection(
            f"devices/{device_id}/appliance_predictions"
        )
        docs = predictions_ref.where("status", "==", "unidentified").stream()

        signatures_data = []
        first_signature_len = None

        for doc in docs:
            data = doc.to_dict()
            signature_array = data.get("signature", [])

            # NEW: Check if signature is a list of maps
            if not isinstance(signature_array, list) or not signature_array:
                logging.warning(f"Skipping signature {doc.id}: 'signature' field is not a list or is empty.")
                continue

            # Extract a single numeric feature (powerWatt) from each map
            signature_numeric = []
            for reading in signature_array:
                if isinstance(reading, dict) and 'powerWatt' in reading and isinstance(reading['powerWatt'], (int, float)):
                    signature_numeric.append(reading['powerWatt'])
                else:
                    logging.warning(f"Skipping signature {doc.id} due to malformed reading data.")
                    # If any single reading is malformed, skip the entire signature
                    signature_numeric = []
                    break
            
            # If the signature was malformed, continue to the next document
            if not signature_numeric:
                continue
            
            # Ensure consistent signature length
            if first_signature_len is None:
                first_signature_len = len(signature_numeric)
            
            if len(signature_numeric) != first_signature_len:
                logging.warning(f"Skipping signature with inconsistent length: {doc.id}")
                continue

            signatures_data.append({
                "id": doc.id,
                "signature": signature_numeric,
                "predicted_label": data.get("predicted_label", None)
            })

        if len(signatures_data) < 2:
            raise HTTPException(
                status_code=404,
                detail="Not enough unidentified signatures to form clusters. At least 2 are required."
            )
        
        if len(signatures_data) < n_clusters:
            n_clusters = len(signatures_data)
        
        signatures_np = np.array([d["signature"] for d in signatures_data], dtype=float)
        signature_ids = [d["id"] for d in signatures_data]
        predicted_labels = [d["predicted_label"] for d in signatures_data]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(signatures_np)
        
        clusters_ref = FIRESTORE_CLIENT.collection(f"devices/{device_id}/clusters")
        timestamp = datetime.utcnow()

        old_clusters = clusters_ref.stream()
        for c in old_clusters:
            c.reference.delete()

        cluster_map = {}
        for i, cluster in enumerate(cluster_ids):
            sig_id = signature_ids[i]
            predicted_label = predicted_labels[i]
            
            cluster_map.setdefault(cluster, {
                "signature_ids": [],
                "predicted_labels": [],
                "signatures_np": []
            })
            cluster_map[cluster]["signature_ids"].append(sig_id)
            cluster_map[cluster]["predicted_labels"].append(predicted_label)
            cluster_map[cluster]["signatures_np"].append(signatures_np[i])

        batch = FIRESTORE_CLIENT.batch()

        for cluster_id, info in cluster_map.items():
            cluster_signatures_np = np.array(info["signatures_np"])
            avg_power = np.mean(cluster_signatures_np) if cluster_signatures_np.size > 0 else 0
            
            most_common_label = Counter(info["predicted_labels"]).most_common(1)
            suggested_label = most_common_label[0][0] if most_common_label and most_common_label[0][0] is not None else "Unnamed Appliance"
            
            new_cluster_ref = clusters_ref.document()
            batch.set(new_cluster_ref, {
                "user_label": suggested_label,
                "status": "unlabeled",
                "summary": {
                    "avg_power": float(avg_power),
                    "count": len(info["signature_ids"]),
                    "signatures": info["signature_ids"]
                },
                "centroid": kmeans.cluster_centers_[cluster_id].tolist(),
                "created_at": timestamp,
            })
            
            for sig_id in info["signature_ids"]:
                sig_ref = predictions_ref.document(sig_id)
                batch.update(sig_ref, {
                    "status": "clustered",
                    "cluster_id": new_cluster_ref.id
                })

        batch.commit()

        return {
            "status": "success",
            "clusters_created": len(cluster_map),
            "device_id": device_id,
            "timestamp": timestamp.isoformat(),
            "message": f"Clustering complete! {len(cluster_map)} new suggested clusters created."
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Clustering failed due to an unexpected error.")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")