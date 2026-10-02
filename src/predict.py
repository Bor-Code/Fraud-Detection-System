import pandas as pd
from typing import Dict, Any, List
from src.persistence import load_artifacts

def predict_single(record: Dict[str, Any]) -> Dict[str, Any]:
    model, scaler, threshold = load_artifacts()
    df = pd.DataFrame([record])
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    prob = float(model.predict_proba(df)[0, 1])
    pred = int(prob >= threshold)
    return {"probability": prob, "prediction": pred}

def predict_batch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model, scaler, threshold = load_artifacts()
    df = pd.DataFrame(records)
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    results = [{"probability": float(p), "prediction": int(c)} for p, c in zip(probs, preds)]
    return results
