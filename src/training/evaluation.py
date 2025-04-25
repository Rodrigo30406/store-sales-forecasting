from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import sqrt
from src.evaluation.metrics import rmsle
import logging
from src.utils import load_object
from src.config import CONFIG
import mlflow

logger = logging.getLogger(CONFIG.logger_name)

def evaluate_model(model, X, y_true):
    logger.info("✅ Evaluando modelo...")
    preds = model.predict(X)
    mae_score = mean_absolute_error(y_true, preds)
    rmse_score = sqrt(mean_squared_error(y_true, preds))
    _, rmsle_score, _ = rmsle(y_true, preds)
    
    # Impresión bonita
    logger.info("📊 Resultados de evaluación:")
    logger.info(f"   🔹 RMSE  : {rmse_score:.4f}")
    logger.info(f"   🔹 MAE   : {mae_score:.4f}")
    logger.info(f"   🔹 RMSLE : {rmsle_score:.4f}")

    #mlflow.log_metric("RMSE", rmse_score)
    #mlflow.log_metric("MAE", mae_score)
    mlflow.log_metric("RMSLE_final", rmsle_score)

    return {
        "RMSLE": rmsle_score,
        "RMSE": rmse_score,
        "MAE": mae_score}