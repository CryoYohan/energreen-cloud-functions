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
from google.cloud import firestore

# ---- Config ----
PROJECT_ID = os.getenv("GCP_PROJECT") or "your-gcp-project"
FIRESTORE_CLIENT = firestore.Client(project=PROJECT_ID)

app = FastAPI()

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:5173", "https://your-production-frontend.com"],  # restrict to your frontend domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request Schema ----
class ClusterRequest(BaseModel):
    device_id: str
    n_clusters: int = 3  # default cluster count


# ---- Endpoint ----
@app.post("/cluster-appliances")
async def cluster_appliances(req: ClusterRequest):
    try:
        device_id = req.device_id
        n_clusters = req.n_clusters

        # 1. Fetch confirmed signatures from Firestore
        predictions_ref = FIRESTORE_CLIENT.collection(
            f"devices/{device_id}/appliance_predictions"
        )
        docs = predictions_ref.where("status", "==", "confirmed").stream()

        signatures = []
        signature_ids = []
        labels = []

        for doc in docs:
            data = doc.to_dict()
            if "signature" in data:
                signatures.append(data["signature"])  # numeric array
                signature_ids.append(doc.id)
                labels.append(data.get("confirmed_label", "Unknown"))

        if not signatures:
            raise HTTPException(status_code=404, detail="No confirmed signatures found")

        X = np.array(signatures, dtype=float)

        # 2. Run clustering (KMeans)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(X)

        # 3. Save cluster results back into Firestore
        clusters_ref = FIRESTORE_CLIENT.collection(f"devices/{device_id}/clusters")
        timestamp = datetime.utcnow()

        # Delete old clusters first (optional, keep history if you want)
        old_clusters = clusters_ref.stream()
        for c in old_clusters:
            c.reference.delete()

        cluster_map = {}
        for sig_id, label, cluster in zip(signature_ids, labels, cluster_ids):
            cluster_map.setdefault(cluster, {"appliances": [], "ids": []})
            cluster_map[cluster]["appliances"].append(label)
            cluster_map[cluster]["ids"].append(sig_id)

        for cluster_id, info in cluster_map.items():
            clusters_ref.add(
                {
                    "cluster_id": cluster_id,
                    "appliances": info["appliances"],
                    "signature_ids": info["ids"],
                    "centroid": kmeans.cluster_centers_[cluster_id].tolist(),
                    "created_at": timestamp,
                }
            )

        return {
            "status": "success",
            "clusters_created": len(cluster_map),
            "device_id": device_id,
            "timestamp": timestamp.isoformat(),
        }

    except Exception as e:
        logging.exception("Clustering failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Cluster NILM API running"}
