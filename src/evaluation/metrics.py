from src.utils.serialization import load_object
import tensorflow.keras.backend as K # type: ignore
import numpy as np

def rmsle_k(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(1 + y_true) - K.log(1 + y_pred))))

def rmsle(y_true, y_pred):
    scaler = load_object("models/scaler_sales.pkl")
    y_true = np.asarray(y_true).astype(np.float64)
    y_pred = np.asarray(y_pred).astype(np.float64)
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    score = np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))
    return "rmsle", score, False