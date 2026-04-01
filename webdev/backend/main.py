from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import datetime

from inference import predict_rul, classify_status

app = FastAPI(title="Universal Predictive Maintenance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated base state for universal machines
# We initialize realistic sensor values close to C-MAPSS average to generate good predictions
base_machines = [
    {
        "id": "UMA-01", 
        "name": "Universal Machine Alpha", 
        "sensors": [
            -0.0007, -0.0004, 100.0, # OpSets 1-3
            518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 
            2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 
            8.4195, 0.03, 392, 2388, 100.00, 39.06, 23.4190
        ]
    },
    {
        "id": "UMB-02", 
        "name": "Universal Machine Beta", 
        "sensors": [
            0.0019, -0.0003, 100.0, 
            518.67, 642.15, 1591.82, 1403.14, 14.62, 21.61, 553.75, 
            2388.04, 9044.07, 1.30, 47.49, 522.28, 2388.07, 8131.49, 
            8.4318, 0.03, 392, 2388, 100.00, 39.00, 23.4236
        ]
    },
    {
        "id": "UMG-03", 
        "name": "Universal Node Gamma", 
        "sensors": [
            -0.0043, 0.0003, 100.0, 
            518.67, 642.35, 1587.99, 1404.20, 14.62, 21.61, 554.26, 
            2388.08, 9052.94, 1.30, 47.27, 522.42, 2388.03, 8133.23, 
            8.4178, 0.03, 390, 2388, 100.00, 38.95, 23.3442
        ]
    }
]

@app.get("/api/machines")
def get_machines():
    results = []
    for m in base_machines:
        # Mutate sensory data slightly to simulate real-time degradation
        # We perturb a few key sensors that change significantly over time
        m["sensors"][4] += random.uniform(-0.1, 0.2)   # Sensor 2
        m["sensors"][5] += random.uniform(-0.5, 1.0)   # Sensor 3
        m["sensors"][6] += random.uniform(-0.5, 1.0)   # Sensor 4
        m["sensors"][11] += random.uniform(-2.0, 5.0)  # Sensor 9
        
        # ML Inference using 24 features
        rul = predict_rul(m["sensors"])
        status = classify_status(rul)
        
        # Prepare abstract diagnostic clusters for the universal UI
        frontend_data = {
            "id": m["id"],
            "name": m["name"],
            "rul_days": rul,
            "status": status,
            # We map generic diagnostic clusters using aggregations of the sensors
            "cluster_alpha": round(m["sensors"][4], 2), # Temperature proxy
            "cluster_beta": round(m["sensors"][5], 1),  # Pressure proxy
            "cluster_gamma": round(m["sensors"][11], 0) # Speed/Flow proxy
        }
        results.append(frontend_data)
        
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
