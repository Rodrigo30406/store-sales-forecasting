from src.ingestion import DataLoader
from src.utils import load_object, save_predictions
from src.config import CONFIG

def run_prediction():
    print("🔄 Cargando datos de test...")
    loader = DataLoader()
    X_test = loader.load_test_parquet()
    indexes = X_test.index.values

    print("🧠 Cargando modelo, scaler y pipeline...")
    model = load_object(CONFIG.get_path("model"))
    scaler = load_object(CONFIG.get_path("scaler"))
    pipeline = load_object(CONFIG.get_path("pipeline"))

    print("⚙️ Aplicando pipeline de transformación...")
    X_test = pipeline.transform(X_test)
    X_test = X_test.drop(columns=["date","sales"])

    print("📈 Generando predicciones...")
    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    print("💾 Guardando resultados...")
    save_predictions(indexes, preds, CONFIG.get_path("predictions"))


if __name__ == "__main__":
    run_prediction()