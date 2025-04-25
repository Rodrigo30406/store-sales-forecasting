from src.training import prepare_data, train_model_lgbm, evaluate_model
from src.utils import save_model
from src.utils.logging import setup_logger
from src.config import CONFIG
import mlflow

logger = setup_logger(CONFIG.logger_name, CONFIG.get_path('train_log'))

def main():
    mlflow.set_experiment("store-sales")
    with mlflow.start_run(run_name="LGBM_Training"):
        X_train, X_val, y_train, y_val = prepare_data()
        model = train_model_lgbm(X_train, X_val, y_train, y_val)
        logger.info(f"✅ Entrenamiento finalizado y modelo guardado en {CONFIG.get_path('model')}")
        evaluate_model(model, X_val, y_val)
    
if __name__ == "__main__":
    main()
