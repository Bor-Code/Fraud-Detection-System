from fastapi import FastAPI, HTTPException
from typing import List
from api.schemas import Transaction, PredictionResponse, BatchPredictionResponse, DriftReportResponse
from src.predict import predict_single, predict_batch
from src.drift import detect_drift
import pandas as pd
from src.config import DATA_PATH

app = FastAPI(title="Fraud Detection API")

import subprocess

try:
    from src.db import save_prediction
except ImportError:
    save_prediction = lambda tx, prob, pred: None

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict/single", response_model=PredictionResponse)
def predict_single_transaction(tx: Transaction) -> PredictionResponse:
    try:
        tx_dict = tx.model_dump()
        res = predict_single(tx_dict)
        save_prediction(tx_dict, res["probability"], res["prediction"])
        return PredictionResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_multiple(txs: List[Transaction]) -> BatchPredictionResponse:
    try:
        records = [tx.model_dump() for tx in txs]
        results = predict_batch(records)
        return BatchPredictionResponse(predictions=[PredictionResponse(**r) for r in results])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drift", response_model=DriftReportResponse)
def check_drift(txs: List[Transaction]) -> DriftReportResponse:
    try:
        df_new = pd.DataFrame([tx.model_dump() for tx in txs])
        df_train = pd.read_csv(DATA_PATH).sample(n=1000)
        features = ["Amount", "Time"]
        report = detect_drift(df_train, df_new, features)
        return DriftReportResponse(report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain_model():
    try:
        # Trigger the retraining pipeline in the background
        subprocess.Popen(["python", "src/train_pipeline.py"])
        return {"status": "Retraining started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
