import pandas as pd
from src.data_loader import load_and_validate
from src.splits import split_data
from src.preprocessing import Preprocessor
from src.tune import tune_xgboost
from src.models import MODEL_REGISTRY
from src.persistence import save_artifacts
from src.evaluate import evaluate_predictions
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading and validating data...")
    df = load_and_validate()
    
    logger.info("Splitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    logger.info("Scaling features...")
    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    
    logger.info("Tuning XGBoost hyperparameters with Optuna...")
    best_params = tune_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, n_trials=10)
    logger.info(f"Best hyperparameters found: {best_params}")
    
    logger.info("Training final model with best hyperparameters...")
    best_model = MODEL_REGISTRY["xgb"](**best_params)
    best_model.fit(X_train_scaled, y_train)
    
    logger.info("Evaluating final model...")
    y_prob = best_model.predict_proba(X_val_scaled)[:, 1]
    y_pred = best_model.predict(X_val_scaled)
    metrics = evaluate_predictions(y_val, y_pred, y_prob)
    logger.info(f"Final Validation Metrics: {metrics}")
    
    logger.info("Saving model and preprocessor artifacts...")
    save_artifacts(best_model, preprocessor, 0.5)
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
