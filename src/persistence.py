import os
import joblib
import boto3
from botocore.exceptions import ClientError
from typing import Any, Tuple
from src.config import MODEL_PATH, SCALER_PATH, THRESHOLD_PATH

MINIO_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
BUCKET_NAME = "models"

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

def init_bucket():
    try:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
    except ClientError as e:
        # If bucket already exists, it's fine
        pass
    except Exception as e:
        print(f"S3 init error (is minio running?): {e}")

def save_artifacts(model: Any, scaler: Any, threshold: float) -> None:
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)
    
    init_bucket()
    try:
        s3_client.upload_file(MODEL_PATH, BUCKET_NAME, "model.joblib")
        s3_client.upload_file(SCALER_PATH, BUCKET_NAME, "scaler.joblib")
        s3_client.upload_file(THRESHOLD_PATH, BUCKET_NAME, "threshold.joblib")
    except Exception as e:
        print(f"Warning: Could not upload models to S3: {e}")

def load_artifacts() -> Tuple[Any, Any, float]:
    try:
        s3_client.download_file(BUCKET_NAME, "model.joblib", MODEL_PATH)
        s3_client.download_file(BUCKET_NAME, "scaler.joblib", SCALER_PATH)
        s3_client.download_file(BUCKET_NAME, "threshold.joblib", THRESHOLD_PATH)
    except Exception as e:
        print(f"Warning: Could not download models from S3 (using local if available): {e}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, scaler, threshold
