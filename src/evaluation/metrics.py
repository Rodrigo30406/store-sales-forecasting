import tensorflow.keras.backend as K # type: ignore

def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(1 + y_true) - K.log(1 + y_pred))))