from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from src.evaluation.metrics import rmsle
import logging
from src.config import CONFIG
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

logger = logging.getLogger(CONFIG.logger_name)

def train_model_lgbm(X_train, X_val, y_train, y_val):

    model_params = CONFIG.get_model_params()
    categorical = CONFIG.get_variable('categorical')
    mlflow.lightgbm.autolog()
    
    model = LGBMRegressor(**model_params)
    
    #mlflow.log_params(model_params)
    mlflow.log_param("model", CONFIG.get_active_model_name())
    logger.info("🚀 Entrenando modelo LightGBM...")
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        categorical_feature=categorical,
        eval_metric= rmsle,
        callbacks=[
            early_stopping(stopping_rounds=1000, min_delta=1e-7),
            log_evaluation(period=1000)
        ]
    )

    # Ejemplo de datos de entrada usados en el modelo
    input_example = X_train.iloc[:5]  # muestra representativa

    # Inferencia de la firma
    signature = infer_signature(X_train, model.predict(X_train))

    #mlflow.sklearn.log_model(
    #    sk_model=model,
    #    artifact_path="model",
    #    signature=signature,
    #    input_example=input_example,
    #    registered_model_name="lgbm_store_sales"
    #)

    return model