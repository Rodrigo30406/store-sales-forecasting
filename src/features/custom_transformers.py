from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class DatePartAdder(BaseEstimator, TransformerMixin):
    def __init__(self, date_column="date"):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column])
        X["day"] = X[self.date_column].dt.day
        X["month"] = X[self.date_column].dt.month
        X["weekday"] = X[self.date_column].dt.weekday
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=["store_nbr", "family"]):
        self.columns = columns
        self.category_maps = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.category_maps[col] = pd.Series(X[col].unique()).sort_values().reset_index(drop=True)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            cat_list = self.category_maps[col]
            cat_to_idx = {cat: idx for idx, cat in enumerate(cat_list)}
            X[col] = X[col].map(cat_to_idx).fillna(-1).astype(int)
        return X

class AdaptiveScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns=["onpromotion", "sales", "day", "month", "weekday"]):
        self.columns = columns
        self.scalers = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                scaler = MinMaxScaler()
                scaler.fit(X[[col]])
                self.scalers[col] = scaler
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.scalers:
                X[col] = self.scalers[col].transform(X[[col]])
        return X

class CyclicEncoder(BaseEstimator, TransformerMixin):
    """
    Transforma columnas cíclicas (como día del mes, mes, día de la semana) 
    en sus componentes seno y coseno para preservar su naturaleza circular.
    
    columns_config: dict donde la clave es el nombre de la columna 
    y el valor es el número máximo del ciclo (ej: 31 para día, 12 para mes).
    """
    def __init__(self, columns_config=None):
        self.columns_config = columns_config if columns_config else {}

    def fit(self, X, y=None):
        return self  # no entrena nada, es determinista

    def transform(self, X):
        X = X.copy()
        for col, period in self.columns_config.items():
            X[f"{col}_sin"] = np.sin(2 * np.pi * X[col] / period)
            X[f"{col}_cos"] = np.cos(2 * np.pi * X[col] / period)
            X.drop(columns=[col], inplace=True)
        return X

class WeekendFlagger(BaseEstimator, TransformerMixin):
    """
    Agrega una columna binaria 'is_weekend' basada en la columna 'weekday'.
    Asume que los valores de 'weekday' están en formato 0=Lunes ... 6=Domingo.
    """
    def __init__(self, weekday_col="weekday"):
        self.weekday_col = weekday_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.weekday_col in X.columns:
            X["is_weekend"] = X[self.weekday_col].isin([5, 6]).astype(int)
        else:
            raise ValueError(f"'{self.weekday_col}' not found in columns.")
        return X