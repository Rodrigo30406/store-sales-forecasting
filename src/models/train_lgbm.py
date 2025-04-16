import pandas as pd
import lightgbm as lgb
import pickle
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.ingestion.load import DataLoader
from src.pipelines.pipeline_builder import build_pipeline
from sklearn.preprocessing import MinMaxScaler
from src.evaluation.metrics import rmsle_np
from src.utils.serialization import load_object, save_object
from src.config import CONFIG
import numpy as np
import os

def main():
    print("📦 Cargando datos...")
    loader = DataLoader()
    df = loader.load_train_parquet()  # ajusta si quieres más/menos

    scaler = MinMaxScaler()
    dummy = scaler.fit_transform(df[["sales"]])
    save_object(scaler, "models/scaler_sales.pkl")

    print("⚙️ Aplicando transformaciones del pipeline...")
    pipeline = build_pipeline()
    df = pipeline.fit_transform(df)

    print("📊 Preparando datos...")
    X = df.drop(columns=["sales","date"])
    y = df["sales"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Entrenando modelo LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=100000,
        learning_rate=0.05,
        objective="regression_l1",
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              categorical_feature=CONFIG.get_variable('categorical'),
              eval_metric= rmsle_np,
              callbacks=[
                         log_evaluation(period=1000)
                        ],)

    print("✅ Evaluando modelo...")
    preds = model.predict(X_val)

    
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmsle_score = rmsle_np(y_val.values, preds)

    print(f"📉 MAE: {mae:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")
    print(f"📉 RMSLE: {rmsle_score:.4f}")

    print("💾 Guardando modelo entrenado...")
    os.makedirs("models", exist_ok=True)
    with open("models/lgbm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Entrenamiento finalizado y modelo guardado en models/lgbm_model.pkl")

if __name__ == "__main__":
    main()
