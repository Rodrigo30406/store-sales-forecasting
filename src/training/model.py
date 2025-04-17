from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from src.evaluation.metrics import rmsle
from src.config import CONFIG

def train_model_lgbm(X_train, X_val, y_train, y_val):

    model = LGBMRegressor(**CONFIG.get_model_params())
    print("🚀 Entrenando modelo LightGBM...")
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        categorical_feature=CONFIG.get_variable('categorical'),
        eval_metric= rmsle,
        callbacks=[
            early_stopping(stopping_rounds=1000, min_delta=1e-7),
            log_evaluation(period=1000)
        ]
    )
    return model