from src.data import DataLoader
from src.utils import load_object, save_predictions, setup_logger
from src.config import CONFIG

logger = setup_logger(CONFIG.logger_name, CONFIG.get_path('predict_log'))

def run_prediction():
    logger.info("🔄 Cargando datos de test...")
    loader = DataLoader()
    X_test = loader.load_test_parquet()
    indexes = X_test.index.values

    logger.info("🧠 Cargando modelo, scaler y pipeline...")
    model = load_object(CONFIG.get_path("model"))
    scaler = load_object(CONFIG.get_path("scaler"))
    pipeline = load_object(CONFIG.get_path("pipeline"))

    logger.info("⚙️ Aplicando pipeline de transformación...")
    X_test = pipeline.transform(X_test)
    X_test = X_test.drop(columns=["date","sales"])

    logger.info("📈 Generando predicciones...")
    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    logger.info("💾 Guardando resultados...")
    save_predictions(indexes, preds, CONFIG.get_path("predictions"))


if __name__ == "__main__":
    run_prediction()