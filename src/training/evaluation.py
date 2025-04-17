from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import sqrt
from src.evaluation.metrics import rmsle
from src.utils import load_object
from src.config import CONFIG

def evaluate_model(model, X, y_true):
    print("✅ Evaluando modelo...")
    preds = model.predict(X)
    mae_score = mean_absolute_error(y_true, preds)
    rmse_score = sqrt(mean_squared_error(y_true, preds))
    _, rmsle_score, _ = rmsle(y_true, preds)
    
    # Impresión bonita
    print("\n📊 Resultados de evaluación:")
    print(f"   🔹 RMSE  : {rmse_score:.4f}")
    print(f"   🔹 MAE   : {mae_score:.4f}")
    print(f"   🔹 RMSLE : {rmsle_score:.4f}")

    return {
        "RMSLE": rmsle_score,
        "RMSE": rmse_score,
        "MAE": mae_score}