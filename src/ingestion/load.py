import pandas as pd
from src.config import CONFIG

class DataLoader:
    def __init__(self, config=CONFIG):
        self.config = config

    def load_train_data(self, n_rows=None):
        return pd.read_csv(self.config.get_path("train"), parse_dates=["date"], nrows=n_rows)

    def load_test_data(self, n_rows=None):
        return pd.read_csv(self.config.get_path("test"), parse_dates=["date"], nrows=n_rows)

    def load_store_metadata(self):
        return pd.read_csv(self.config.get_path("store_metadata"))
    
    def load_train_parquet(self, columns=None):
        """
        Carga el archivo de entrenamiento en formato Parquet.
        Parámetros:
            columns (list, opcional): columnas a cargar del Parquet (para optimizar memoria)
        """
        path = self.config.get_path("train_parquet")
        return pd.read_parquet(path, columns=columns)