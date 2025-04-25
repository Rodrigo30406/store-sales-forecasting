import logging
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.data import DataLoader
from src.pipelines import PipelineBuilder
from src.utils import save_object, log_pipeline_readable
from src.config import CONFIG


logger = logging.getLogger(CONFIG.logger_name)

def prepare_data():
    loader = DataLoader()
    logger.info("📦 Cargando datos...")
    df = loader.load_train_parquet()
    logger.info("📊 Preparando datos...")
    X = df.drop(columns=[CONFIG.get_variable("target")])
    y = df[CONFIG.get_variable("target")]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("⚙️ Aplicando transformaciones del pipeline...")
    X_train, X_val = transform_data(X_train, X_val)
    X_train = X_train.drop(columns=["date"])
    X_val = X_val.drop(columns=["date"])
    y_train, y_val = scale_target(y_train, y_val)
    return X_train, X_val, y_train, y_val

def transform_data(X_train, X_val=None):
    builder = PipelineBuilder()
    pipeline = builder.build_preprocessor_pipeline()
    X_train = pipeline.fit_transform(X_train)
    mlflow.sklearn.log_model(pipeline, artifact_path="preprocessing_pipeline")
    log_pipeline_readable(pipeline)
    if X_val is not None:
        X_val = pipeline.transform(X_val)
        return X_train, X_val
    return X_train

def scale_target(y_train, y_val):
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val = scaler.transform(y_val.values.reshape(-1, 1))
    mlflow.sklearn.log_model(scaler, artifact_path="scaler")
    save_object(scaler, CONFIG.get_path('scaler'))
    return y_train, y_val