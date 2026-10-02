# Fraud Detection System

## Summary
End to end machine learning system for credit card fraud detection. It includes data pipelines, multiple models, hyperparameter tuning, model explainability, drift detection, and an API.

## Architecture
- Data Loader & Preprocessing
- Stratified Splits & Feature Engineering
- Imbalance Handling (SMOTE, Undersampling)
- Training Pipeline (LR, RF, XGB, LGBM, MLP)
- MLflow Tracking & Optuna Tuning
- FastAPI for Endpoints
- Streamlit Dashboard

## Installation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Usage
Start API:
uvicorn api.main:app --reload

Start Dashboard:
streamlit run dashboard/app.py

## API Examples
POST /predict
{
  "Time": 0.0,
  "Amount": 100.0,
  "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0, "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
  "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
  "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0
}

## Results
| Model | Sampler | PR-AUC |
|-------|---------|--------|
| TBD   | TBD     | TBD    |

## Development Workflow
Feature branching with PRs, automated CI testing via GitHub Actions.
